"""Local evidence proposals, not measured attention or verified morphology."""
import hashlib
import json
import math


PROMPT_VERSION = "evidence-boxes-v2-no-coordinate-example-20261104"


def grounding_prompt(question, maximum=5, coordinate_space="normalized", image_size=None):
    prompt = (
        "Locate visible tissue findings relevant to the question in THIS image. "
        "Return only one JSON object, without reasoning or markdown. "
        "Do not infer coordinates from slide identifiers or external knowledge. "
        "Coordinates must be normalized to this supplied image: x=0 left, x=1 right, "
        "y=0 top, y=1 bottom. Each bbox is [left,top,right,bottom]. "
        "Use tight boxes around actual visible morphology, not the whole image. "
        "Do not identify a cell when the resolution does not support it. "
        f"Return zero to {maximum} distinct regions. If you cannot locate supporting "
        "morphology, return an empty regions array and explain in no_evidence_reason. "
        "Set primary_id to the most relevant region ID, or null if none. "
        "The region list is diagnostic priority, NOT temporal gaze order. "
        "The object must have keys regions (array), primary_id (string or null), "
        "and no_evidence_reason (string). Each region must have id (unique string), "
        "bbox (four normalized numbers), and finding (specific visible finding string). "
        "Calculate coordinates from the supplied image. "
        f"Question: {question}"
    )
    if coordinate_space == "input_pixels":
        width,height=image_size
        prompt=prompt.replace("Coordinates must be normalized to this supplied image: x=0 left, x=1 right, y=0 top, y=1 bottom.",
                              f"Coordinates are absolute pixels on this supplied {width} by {height} image, with origin at the upper-left. x ranges from 0 to {width}; y from 0 to {height}.")
        prompt=prompt.replace("four normalized numbers","four pixel coordinate numbers")
        prompt+=f' Also return coordinate_space="input_pixels" and image_size=[{width},{height}]. These metadata must match the supplied canvas. Avoid full-image boxes; localize specific tissue structures.'
    elif coordinate_space != "normalized":
        raise ValueError("Unknown coordinate convention")
    return prompt


def parse_evidence(text, maximum=5):
    decoder = json.JSONDecoder()
    payload = None
    # A model may prepend text; retain it as raw output but parse only full JSON.
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "regions" in value:
            payload = value
            break
    if payload is None:
        raise ValueError("No complete evidence JSON object")
    regions = payload["regions"]
    if not isinstance(regions, list) or len(regions) > maximum:
        raise ValueError("regions must be an array within the configured limit")
    ids = set()
    cleaned = []
    for region in regions:
        if not isinstance(region, dict):
            raise ValueError("Region must be an object")
        identifier, bbox, finding = region.get("id"), region.get("bbox"), region.get("finding")
        if not isinstance(identifier,str) or not identifier or identifier in ids:
            raise ValueError("Missing/duplicate region ID")
        if not isinstance(bbox,list) or len(bbox) != 4 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in bbox):
            raise ValueError("Invalid bounding-box coordinates")
        left, top, right, bottom = map(float,bbox)
        if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
            raise ValueError("Out-of-range or reversed bounding box; no clipping applied")
        if not isinstance(finding,str) or not finding.strip():
            raise ValueError("A localized finding description is required")
        cleaned.append({"id":identifier,"bbox_norm":[left,top,right,bottom],"finding":finding.strip(),
                        "review_flags":["broad_box"] if (right-left)*(bottom-top)>0.5 else []})
        ids.add(identifier)
    primary = payload.get("primary_id")
    if regions and primary not in ids:
        raise ValueError("primary_id does not identify an existing region")
    if not regions and primary is not None:
        raise ValueError("Empty evidence must have primary_id=null")
    reason = payload.get("no_evidence_reason","")
    if not regions and (not isinstance(reason,str) or not reason.strip()):
        raise ValueError("Empty evidence must state why evidence is unavailable")
    return {"regions":cleaned,"primary_id":primary,"no_evidence_reason":reason,
            "schema_valid":True,"clinical_validity":"unreviewed"}


def parse_pixel_evidence(text, image_size, maximum=5):
    """Explicit pixel protocol only; never infer units from coordinate magnitudes."""
    decoder=json.JSONDecoder();payload=None
    for index,char in enumerate(text):
        if char!="{":continue
        try:value,_=decoder.raw_decode(text[index:])
        except json.JSONDecodeError:continue
        if isinstance(value,dict) and "regions" in value:payload=value;break
    if payload is None:raise ValueError("No complete evidence JSON object")
    if payload.get("coordinate_space")!="input_pixels" or payload.get("image_size")!=list(image_size):
        raise ValueError("Explicit pixel coordinate space and matching canvas size required")
    width,height=image_size
    if not isinstance(payload["regions"],list):raise ValueError("regions must be an array")
    converted=[]
    for region in payload["regions"]:
        box=region.get("bbox") if isinstance(region,dict) else None
        if not isinstance(box,list) or len(box)!=4 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in box):
            raise ValueError("Invalid pixel bounding box")
        converted.append({**region,"bbox":[box[0]/width,box[1]/height,box[2]/width,box[3]/height]})
    result=parse_evidence(json.dumps({**payload,"regions":converted}),maximum)
    result.update(coordinate_space="input_pixels",canvas_size=list(image_size),raw_pixel_regions=payload["regions"])
    return result


def map_regions(evidence, roi):
    """Affine mapping from original image-normalized boxes to WSI level-0."""
    values = [roi[k] for k in ("x0","y0","width0","height0")]
    if any(not math.isfinite(v) for v in values) or min(values[2:]) <= 0:
        raise ValueError("Invalid ROI geometry")
    result = {**evidence,"regions":[]}
    for region in evidence["regions"]:
        l,t,r,b=region["bbox_norm"]
        box=[roi["x0"]+l*roi["width0"],roi["y0"]+t*roi["height0"],
             roi["x0"]+r*roi["width0"],roi["y0"]+b*roi["height0"]]
        result["regions"].append({**region,"bbox_level0":box,
                                 "point_level0":[(box[0]+box[2])/2,(box[1]+box[3])/2],
                                 "is_primary":region["id"]==evidence["primary_id"]})
    return result


def infer_evidence(image, question, model, processor, maximum=5, max_new_tokens=1024,coordinate_space="normalized"):
    import torch
    from qwen_vl_utils import process_vision_info
    prompt=grounding_prompt(question,maximum,coordinate_space,image.size)
    messages=[{"role":"system","content":"You localize visible image evidence. Output strict JSON only."},
              {"role":"user","content":[{"type":"image","image":image},{"type":"text","text":prompt}]}]
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    images,_=process_vision_info(messages)
    inputs=processor(text=[text],images=images,padding=True,return_tensors="pt").to(model.device)
    processed_size=None
    if coordinate_space=="input_pixels":
        grid=inputs["image_grid_thw"][0].tolist()
        patch=int(model.config.vision_config.patch_size)
        processed_size=[int(grid[2])*patch,int(grid[1])*patch]
        if processed_size!=list(image.size):raise ValueError("Processor changed explicit grounding canvas dimensions")
    with torch.inference_mode():
        ids=model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False)
    raw=processor.batch_decode([out[len(inp):] for inp,out in zip(inputs.input_ids,ids)],
                               skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip()
    receipt={"raw_output":raw,"prompt_version":PROMPT_VERSION if coordinate_space=="normalized" else "evidence-boxes-v3-explicit-pixel-canvas-20261104",
             "prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),
             "input_tokens":int(inputs.input_ids.shape[-1]),
             "output_tokens":int(ids.shape[-1]-inputs.input_ids.shape[-1]),
             "max_new_tokens":max_new_tokens}
    receipt.update(coordinate_space_requested=coordinate_space,input_canvas_size=list(image.size),processed_canvas_size=processed_size)
    try:
        receipt.update(parse_pixel_evidence(raw,image.size,maximum) if coordinate_space=="input_pixels" else parse_evidence(raw,maximum))
    except ValueError as error:
        receipt.update(schema_valid=False,parse_error=str(error),regions=[],primary_id=None,
                       clinical_validity="unreviewed")
    return receipt

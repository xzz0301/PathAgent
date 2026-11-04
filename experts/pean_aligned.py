"""PEAN adapter for the established observation/qualification/compression rules."""
import csv,json,math
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image,ImageDraw
from .observations import Config,simplify,filter_observations
from .compression import CompressionConfig,compress


def align_samples(rows,slide_size,fps,mode='frame_order',threshold=150.,do_compress=True,grouping_config=None):
    if mode not in ('frame_order','fps_estimate'):raise ValueError('Unknown clock mode')
    if not math.isfinite(fps) or fps<=0:raise ValueError('Invalid header FPS')
    states=[];events=[];previous=None
    for i,row in enumerate(rows):
        valid=bool(row.get('valid_spatial_sample',not row.get('exclusion_reason')))
        ds=row.get('downsample')
        state=None
        if ds and row.get('display_width_level0') and row.get('display_height_level0'):
            state=dict(left=row['display_left_level0'],top=row['display_top_level0'],width=row['display_width_level0'],height=row['display_height_level0'],relative_zoom=1/ds)
        if state is None:
            valid=False
            state=previous or dict(left=0.,top=0.,width=float(slide_size[0]),height=float(slide_size[1]),relative_zoom=1.)
        states.append(state);previous=state
        start=i*1000/fps if mode=='fps_estimate' else float(i)
        end=(i+1)*1000/fps if mode=='fps_estimate' and i<len(rows)-1 else start
        x=row.get('gaze_x_level0');y=row.get('gaze_y_level0')
        x=float(x) if x is not None and math.isfinite(x) else 0.;y=float(y) if y is not None and math.isfinite(y) else 0.
        events.append(dict(event_id=i,frame_index=row['frame_index'],input_valid=valid,x_px=x,y_px=y,x_norm=x/slide_size[0],y_norm=y/slide_size[1],
                           start_time=start,end_time=end,video_s=None,viewport_state_index=i,viewer_relative_zoom=state['relative_zoom'],
                           support_intervals=[[start,end]] if valid and mode=='fps_estimate' and end>start else [],
                           source_exclusion_reason=row.get('exclusion_reason','')))
    result=simplify(events,states,grouping_config or Config())
    result=filter_observations(result,threshold if mode=='fps_estimate' else 0.)
    for s in result['steps']:s['action_linked']=False
    before={'regions':len(result['regions']),'visits':len(result['steps'])}
    if do_compress:result=compress(result,CompressionConfig())
    # Display labels are created after every selection; preserve original IDs.
    ids={r['region_id']:i+1 for i,r in enumerate(result['regions'])}
    for r in result['regions']:r.update(source_region_id=r['region_id'],region_id=ids[r['region_id']])
    for s in result['steps']:
        s.update(source_region_id=s['region_id'],region_id=ids[s['region_id']],
                 valid_samples=len(s['source_event_ids']),start_frame=min(s['source_event_ids']),end_frame=max(s['source_event_ids']),
                 valid_support_ms=None,estimated_support_ms=s['valid_support'] if mode=='fps_estimate' else None,
                 action_evidence='unavailable_no_timestamped_mouse_log')
    for edge in result['edges']:
        edge['from_region']=ids[edge['from_region']];edge['to_region']=ids[edge['to_region']]
    event_steps={event:s['step'] for s in result['steps'] for event in s['source_event_ids']}
    for e in result['events']:
        e.update(source_region_id=e['region_id'],display_region_id=ids.get(e['region_id']),
                 display_step=event_steps.get(e['event_id']),display_retained=e['event_id'] in event_steps)
    result.update(time_basis='uncalibrated_frame_order' if mode=='frame_order' else 'header_fps_estimate_unvalidated',
                  numbering_contract='regions/steps/edges use final display IDs; events and audits retain original region_id with explicit display_* fields',
                  spatial_reference='screen_extent_projection_not_application_viewport_validated',
                  duration_filter_applied=mode=='fps_estimate',min_estimated_support_ms=threshold if mode=='fps_estimate' else None,
                  clock_assumption=None if mode=='frame_order' else 'constant header FPS; valid sample supports its next frame interval; final sample has zero assumed support',
                  measured_time_available=False,operation_evidence_available=False,before_compression=before)
    if mode=='frame_order':
        # No internal union-length placeholder is exposed as measured dwell.
        for group in ('steps','precompression_steps','visits_audit','compression_audit'):
            for s in result.get(group,[]):
                if 'valid_support' in s:s['valid_support']=None
    return result


def ndpa_metadata(path,slide):
    result={'file':str(path),'status':'missing','timestamped_operations':False,'annotations':[]}
    if not path.exists():return result
    root=ET.parse(path).getroot();properties=slide.properties
    required=['openslide.mpp-x','openslide.mpp-y','hamamatsu.XOffsetFromSlideCentre','hamamatsu.YOffsetFromSlideCentre']
    calibrated=all(k in properties for k in required)
    for view in root.findall('.//ndpviewstate'):
        for annotation in view.findall('annotation'):
            nm=[[float(p.findtext('x')),float(p.findtext('y'))] for p in annotation.findall('./pointlist/point')]
            xy=[]
            if calibrated:
                mx,my=(float(properties['openslide.mpp-'+a])*1000 for a in ('x','y'))
                ox,oy=(float(properties['hamamatsu.'+a+'OffsetFromSlideCentre']) for a in ('X','Y'))
                xy=[[(x-ox)/mx+slide.dimensions[0]/2,(y-oy)/my+slide.dimensions[1]/2] for x,y in nm]
            result['annotations'].append({'type':annotation.get('type'),'display_name':annotation.get('displayname'),'closed':annotation.findtext('closed')=='1',
                                          'points_nanometers':nm,'points_level0':xy if calibrated else None,
                                          'inside_fraction':sum(0<=x<slide.dimensions[0] and 0<=y<slide.dimensions[1] for x,y in xy)/len(xy) if xy else None,
                                          'title':view.findtext('title'),'details':view.findtext('details')})
    result.update(status='static_annotations_only',spatial_conversion='scanner_offset_and_mpp' if calibrated else 'missing_scanner_calibration',
                  interpretation='Static annotation geometry; no click/drawing time, no automatic mouse-evidence protection')
    return result


def render(overview,dimensions,result,path,title,objective):
    from .pean import font
    steps=result['steps'];regions=result['regions'];w,h=dimensions
    height=max(1500,300+len(steps)*42)
    page=Image.new('RGB',(2400,height),'white');draw=ImageDraw.Draw(page)
    draw.text((35,25),title,font=font(33),fill='#203d50')
    clock='Frame-ordered route | timing uncalibrated' if not result['duration_filter_applied'] else f"Exploratory {result['min_estimated_support_ms']:g} ms gate | header-FPS estimate"
    draw.text((35,77),clock,font=font(24),fill='#735718')
    scale=min(1430/w,1260/h);size=(round(w*scale),round(h*scale));offset=(35+(1430-size[0])//2,150+(1260-size[1])//2)
    page.paste(overview.resize(size,Image.Resampling.LANCZOS),offset)
    pos={r['region_id']:(offset[0]+r['x_px']*scale,offset[1]+r['y_px']*scale) for r in regions}
    for edge in result['edges']:
        a,b=pos[edge['from_region']],pos[edge['to_region']];length=math.dist(a,b)
        if length<28:continue
        ux,uy=(b[0]-a[0])/length,(b[1]-a[1])/length;end=(b[0]-ux*13,b[1]-uy*13)
        draw.line((a,end),fill='white',width=7);draw.line((a,end),fill='#246590',width=4)
        draw.polygon([end,(end[0]-ux*17+uy*7,end[1]-uy*17-ux*7),(end[0]-ux*17-uy*7,end[1]-uy*17+ux*7)],fill='#246590')
    occupied=[]
    for r in regions:
        x,y=pos[r['region_id']];draw.ellipse((x-11,y-11,x+11,y+11),fill='#246590',outline='white',width=2)
        label=f"R{r['region_id']:02d}";tw=draw.textbbox((0,0),label,font=font(22))[2]+10
        options=[(x+15,y-18),(x+15,y+12),(x-tw-15,y-18),(x-tw-15,y+12),(x+15,y-55),(x+15,y+45)]
        for lx,ly in options:
            lx=max(offset[0],min(lx,offset[0]+size[0]-tw));ly=max(130,min(ly,height-65));box=(lx,ly,lx+tw,ly+31)
            if not any(box[0]<b[2] and box[2]>b[0] and box[1]<b[3] and box[3]>b[1] for b in occupied):break
        occupied.append(box);draw.line((x,y,(box[0]+box[2])/2,(box[1]+box[3])/2),fill='#246590',width=1)
        draw.rectangle(box,fill='white');draw.text((box[0]+5,box[1]),label,font=font(22),fill='#203d50')
    draw.text((1510,150),'Visit   Region   Frames        Samples   Mag. / behaviour',font=font(21),fill='#203d50')
    for i,s in enumerate(steps):
        y=198+i*42
        if i%2==0:draw.rectangle((1495,y-4,2370,y+35),fill='#f0f5f8')
        zoom='/'.join(f'{objective*z:g}' for z in s['zoom_sequence'])
        if len(zoom)>21:zoom=f"{objective*s['zoom_sequence'][0]:g}..{objective*s['zoom_sequence'][-1]:g}"
        text=f"{s['step']:02d}       R{s['region_id']:02d}       {s['start_frame']}-{s['end_frame']}       {s['valid_samples']}       {zoom}x / {s['kind']}"
        draw.text((1510,y),text,font=font(20),fill='#203d50')
    draw.text((35,height-42),'Fixed gaze anchors; arrows indicate visit order. No timestamped mouse events available.',font=font(22),fill='#526572')
    page.save(path,dpi=(300,300))


def run_aligned(raw_root,output):
    import openslide
    from .pean import inventory,read_epr,project_frame,tissue_mask,dump,table,render_quality_control
    output=Path(output);output.mkdir(parents=True,exist_ok=False);audit=inventory(raw_root);dump(output/'inventory.json',audit)
    selected=[]
    for disease in ('bcc','scc','melanoma'):
        candidates=[r for r in audit['records'] if r['identity_verified'] and disease in r['matched_slide']['filename'].lower()]
        selected.append(sorted(candidates,key=lambda r:r['source_file'])[0])
    summaries=[]
    for record in selected:
        info=record['matched_slide'];name=Path(info['filename']).stem;folder=output/name;folder.mkdir()
        metadata=read_epr(record['path']);frames=metadata.pop('frames');w,h=info['dimensions']
        with openslide.OpenSlide(info['path']) as slide:
            overview=slide.get_thumbnail((2400,2400)).convert('RGB')
            annotations=ndpa_metadata(Path(info['path']+'.ndpa'),slide)
        dump(folder/'ndpa_annotations.json',annotations);dump(folder/'recording_metadata.json',metadata)
        overview.save(folder/'overview.png');mask=tissue_mask(overview);rows=[]
        for i,frame in enumerate(frames):
            row=project_frame(frame,metadata,info['level_downsamples'])
            row.update(frame_index=i,recording_id=record['content_id'],slide_id=name,fps_derived_time_ms_unvalidated=i*1000/metadata['fps_header'])
            row['nominal_magnification']=float(info['objective'])/row['downsample'] if info['objective'] and row['downsample'] else None
            if not row['exclusion_reason']:
                ix=min(mask.shape[1]-1,int(row['gaze_x_level0']/w*mask.shape[1]));iy=min(mask.shape[0]-1,int(row['gaze_y_level0']/h*mask.shape[0]))
                if not mask[iy,ix]:row['exclusion_reason']='background_at_overview_resolution'
            row['valid_spatial_sample']=not bool(row['exclusion_reason']);rows.append(row)
        table(folder/'samples.csv',rows);render_quality_control(overview,mask,rows,(w,h),folder/'coordinate_qc.png',name)
        summary={'slide_id':name,'samples':len(rows),'valid_samples':sum(r['valid_spatial_sample'] for r in rows),'variants':{}}
        for mode,subdir in [('frame_order','spatial'),('fps_estimate','fps_estimate_150ms')]:
            dest=folder/subdir;dest.mkdir();result=align_samples(rows,(w,h),metadata['fps_header'],mode)
            result.update(slide_id=name,slide_dimensions=[w,h],source_sha256=metadata['source_sha256'],tail=metadata['tail'])
            dump(dest/'result.json',result)
            visits=[{k:s[k] for k in ('step','region_id','source_region_id','start_frame','end_frame','valid_samples','valid_support_ms','estimated_support_ms','kind','zoom_sequence','action_evidence')} for s in result['steps']]
            table(dest/'visits.csv',visits);table(dest/'regions.csv',result['regions']);table(dest/'compression_audit.csv',result.get('compression_audit',[]))
            render(overview,(w,h),result,dest/'trajectory.png',name,float(info['objective'] or 1))
            summary['variants'][subdir]={'regions':len(result['regions']),'visits':len(result['steps']),'before_compression':result['before_compression'],'duration_filter_applied':result['duration_filter_applied'],'measured_time_available':False}
        summaries.append(summary)
    gaps={'timestamp_unit':'unresolved; raw timestamps and 50FPS disagree','mouse_events':'absent: no MAPDATA in the 90 unique EPR records; no timestamped click/drawing log',
          'viewport':'screen projection per format documentation; UI content bounds not recorded','fixation':'raw samples, no native fixation intervals/confidence/calibration error',
          'annotation':'NDPA is static geometry; TAIL diagnosis is case-level without timing','identity':'90 unique EPR contents across 150 files; no verified individual doctor identity',
          'policy':'Primary spatial output has no duration claim. 150ms variant uses explicit unvalidated header-FPS estimates only. No invented mouse event or fallback focus point.'}
    dump(output/'data_gaps.json',gaps);dump(output/'summary.json',summaries)
    cards=''
    for s in summaries:
        n=s['slide_id'];cards+=f'<section><h2>{n}</h2><p><a href="{n}/samples.csv">原始逐帧表</a> · <a href="{n}/recording_metadata.json">记录头部和尾部</a> · <a href="{n}/ndpa_annotations.json">静态标注解析</a> · <a href="{n}/coordinate_qc.png">坐标与组织筛选</a></p>'
        for variant,title in [('spatial','原规则空间归并与展示压缩（时间未校准）'),('fps_estimate_150ms','50 FPS估计下的150ms敏感性结果（非实测时长）')]:
            v=s['variants'][variant];cards+=f'<h3>{title}</h3><p>{v["regions"]}个区域，{v["visits"]}次访问；压缩前{v["before_compression"]["visits"]}次访问。</p><a href="{n}/{variant}/trajectory.png"><img src="{n}/{variant}/trajectory.png"></a><p><a href="{n}/{variant}/visits.csv">访问表</a> · <a href="{n}/{variant}/result.json">完整结果与假设</a> · <a href="{n}/{variant}/compression_audit.csv">压缩依据</a></p>'
        cards+='</section>'
    (output/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PEAN原规则适配</title><style>body{max-width:1550px;margin:35px auto;padding:0 25px;font:18px/1.8 sans-serif;color:#203d50;background:#f4f7fa}section{background:white;padding:25px;margin:25px 0}img{max-width:100%}a{color:#166891}</style><h1>PEAN：复用既定专家轨迹处理规则</h1><p>固定首点锚点、连续缩放归并、细览与回访匹配、有效支持资格、证据保护几何压缩。缺少可靠时钟与鼠标时序，不虚构操作高亮。两版都按完整WSI显示轨迹，筛选压缩后连续编号。</p><p><a href="data_gaps.json">数据缺口与处理政策</a></p>'+cards+'</html>',encoding='utf-8')
    print(json.dumps(summaries,ensure_ascii=False,indent=2))

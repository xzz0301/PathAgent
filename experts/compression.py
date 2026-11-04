# Shared established rules. Time-axis units are declared by the adapter; no implicit milliseconds.
"""Evidence-constrained geometric representation, retaining original visits in audit.

    This reduces the polyline, not the validity of measured gaze observations.
    There is no target node count and no deletion solely for absence of mouse actions.
"""
from copy import deepcopy
from dataclasses import dataclass,asdict
import math
from .primitives import segment_distance


@dataclass(frozen=True)
class CompressionConfig:
    radius_x_px:float=1500.0
    radius_y_px:float=2600.0
    tolerance:float=1.0
    protect_support:float=1000.0


def select_indices(points,protected,tolerance=1.0):
    if not points:return []
    kept=set(protected)|{0,len(points)-1}
    stack=list(zip(sorted(kept),sorted(kept)[1:]))
    while stack:
        lo,hi=stack.pop()
        if hi-lo<2:continue
        dist,k=max((segment_distance(points[j],points[lo],points[hi]),j) for j in range(lo+1,hi))
        # A path can be geometrically collinear yet travel backwards. Preserve
        # a reversal larger than tolerance instead of silently erasing a revisit.
        if dist<=tolerance:
            ax,ay=points[lo];bx,by=points[hi];length=math.hypot(bx-ax,by-ay)
            if length>0:
                peak=0.0;peak_index=lo
                for j in range(lo+1,hi+1):
                    projection=((points[j][0]-ax)*(bx-ax)+(points[j][1]-ay)*(by-ay))/length
                    if peak-projection>tolerance:
                        k=peak_index if lo<peak_index<hi else j
                        dist=tolerance+1;break
                    if projection>peak:peak,peak_index=projection,j
        if dist>tolerance:
            kept.add(k);stack.extend([(lo,k),(k,hi)])
    return sorted(kept)


def compress(result,cfg=CompressionConfig()):
    if min(cfg.radius_x_px,cfg.radius_y_px,cfg.tolerance,cfg.protect_support)<=0:raise ValueError('Nonpositive compression configuration')
    source=deepcopy(result);steps=source['steps'];regions={r['region_id']:r for r in source['regions']}
    points=[(regions[s['region_id']]['x_px']/cfg.radius_x_px,regions[s['region_id']]['y_px']/cfg.radius_y_px) for s in steps]
    protected={i for i,s in enumerate(steps) if s['action_linked'] or s['valid_support']>=cfg.protect_support}
    kept=select_indices(points,protected,cfg.tolerance)
    selected=[deepcopy(steps[i]) for i in kept];selected_regions={s['region_id'] for s in selected}
    out=deepcopy(source)
    out['precompression_steps']=deepcopy(steps)
    out['compression_audit']=[];reason={}
    for i,s in enumerate(steps):
        why=('operation_evidence' if s['action_linked'] else 'long_valid_observation' if i in protected else
             'endpoint' if i in (0,len(steps)-1) else 'geometric_turn' if i in kept else 'represented_by_bounded_segment')
        reason[s['visit_id']]=why
        row={'source_step':s['step'],'visit_id':s['visit_id'],'source_region_id':s['region_id'],
             'retained':i in kept,'reason':why,'valid_support':s['valid_support'],
             'action_linked':s['action_linked'],'zoom_sequence':s['zoom_sequence']}
        if i not in kept:
            lo=max(k for k in kept if k<i);hi=min(k for k in kept if k>i)
            error=segment_distance(points[i],points[lo],points[hi])
            assert error<=cfg.tolerance+1e-9
            row.update(bracket_source_steps=[steps[lo]['step'],steps[hi]['step']],normalized_error=error)
        out['compression_audit'].append(row)
    seen=set()
    for n,s in enumerate(selected,1):
        s.update(precompression_step=s['step'],step=n,revisit=s['region_id'] in seen,
                 compression_keep_reason=reason[s['visit_id']])
        if s['kind']!='zoom_change':s['kind']='revisit' if s['revisit'] else 'observation'
        seen.add(s['region_id'])
    out['steps']=selected
    out['regions']=[r for r in out['regions'] if r['region_id'] in selected_regions]
    for r in out['regions']:
        retained=[s for s in selected if s['region_id']==r['region_id']]
        # Anchors stay at measured baseline positions. Do not move the point or
        # add removed dwell to the retained visit's evidence.
        r['steps']=[s['step'] for s in retained]
        r['action_linked']=any(s['action_linked'] for s in retained)
        r['action_visit_steps']=[s['step'] for s in retained if s['action_linked']]
    out['edges']=[]
    source_edges={(e['from_step'],e['to_step']):e for e in source['edges']}
    for a,b in zip(selected,selected[1:]):
        lo=a['precompression_step'];hi=b['precompression_step']
        skipped=[deepcopy(s) for s in steps if lo<s['step']<hi]
        edge=source_edges.get((lo,hi))
        out['edges'].append({'from_step':a['step'],'to_step':b['step'],
                             'from_region':a['region_id'],'to_region':b['region_id'],
                             'style':'dashed' if skipped or edge is None or edge['style']=='dashed' else 'solid',
                             'compressed_source_steps':[s['step'] for s in skipped],
                             'compressed_observations':skipped,
                             'meaning':'ordered compressed summary; not a measured saccade'})
    out['compression_config']=asdict(cfg)
    out['compression_summary']={'before_regions':len(regions),'before_visits':len(steps),
                                'after_regions':len(out['regions']),'after_visits':len(selected),
                                'protected_operation_visits':sum(s['action_linked'] for s in steps),
                                'protected_long_observation_visits':sum(s['valid_support']>=cfg.protect_support for s in steps),
                                'all_operation_visits_retained':all(i in kept for i,s in enumerate(steps) if s['action_linked']),
                                'max_removed_normalized_error':max([r.get('normalized_error',0) for r in out['compression_audit']],default=0)}
    return out

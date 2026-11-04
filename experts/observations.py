# Shared established rules. Time-axis units are declared by the adapter; no implicit milliseconds.
"""Region observations supported by gaze, with fixed spatial references.

Viewer state supplies context, never a replacement gaze position.  All markers
are the first supported, on-tissue gaze sample of their region.  No dwell-time
threshold splits observations, and invalid samples never contribute evidence.
"""
from dataclasses import asdict, dataclass
from copy import deepcopy
import math
from .primitives import interval_union


@dataclass(frozen=True)
class Config:
    overview_radius_x_px: float = 1500.0
    overview_radius_y_px: float = 2600.0
    detailed_short_side_px: float = 35000.0
    same_view_iou: float = 0.65
    revisit_view_iou: float = 0.75
    max_reference_pan_fraction: float = 0.20
    zoom_containment: float = 0.85
    zoom_tolerance: float = 0.02


def box(state):
    return [float(state['left']), float(state['top']),
            float(state['left'] + state['width']), float(state['top'] + state['height'])]


def area(b):
    return max(0, b[2]-b[0])*max(0, b[3]-b[1])


def intersection(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0]))*max(0, min(a[3], b[3])-max(a[1], b[1]))


def iou(a, b):
    inter = intersection(a, b)
    return inter / max(1e-12, area(a)+area(b)-inter)


def contains(b, e):
    return b[0] <= e['x_px'] <= b[2] and b[1] <= e['y_px'] <= b[3]


def near_zoom(a, b, tol):
    return abs(a-b)/min(a, b) <= tol+1e-12


def small_pan(a, b, cfg, revisit=False):
    # Compare against the first reference, not against the previous event.
    dx = abs((a[0]+a[2]-b[0]-b[2])/2)
    dy = abs((a[1]+a[3]-b[1]-b[3])/2)
    return (iou(a, b) >= (cfg.revisit_view_iou if revisit else cfg.same_view_iou)
            and dx <= cfg.max_reference_pan_fraction*(a[2]-a[0])
            and dy <= cfg.max_reference_pan_fraction*(a[3]-a[1]))


def local_distance(e, anchor, cfg):
    return math.hypot((e['x_px']-anchor['x_px'])/cfg.overview_radius_x_px,
                      (e['y_px']-anchor['y_px'])/cfg.overview_radius_y_px)


def display_episodes(states, cfg=Config()):
    """Fixed-window pans and nested zoom corrections; independent of dwell."""
    episodes = []
    for idx, state in enumerate(states):
        b, z = box(state), float(state['relative_zoom'])
        current = episodes[-1] if episodes else None
        previous_zoom = float(states[idx-1]['relative_zoom']) if idx else z
        changed = not near_zoom(previous_zoom, z, cfg.zoom_tolerance)
        join = False
        if current:
            ref = current['reference_box']
            if changed and current['has_zoom']:
                # Smallest content window stays the spatial reference during
                # zoom overshoot/correction. Direction need not be monotonic.
                join = intersection(ref, b)/max(1e-12, min(area(ref), area(b))) >= cfg.zoom_containment
            elif not changed:
                join = small_pan(ref, b, cfg)
        if not join:
            current = {'display_episode_id': len(episodes)+1, 'state_indices': [],
                       'reference_box': b, 'zoom_sequence': [previous_zoom, z] if changed else [z],
                       'has_zoom': changed}
            episodes.append(current)
        else:
            if changed:
                current['zoom_sequence'].append(z)
                if area(b) < area(current['reference_box']):
                    current['reference_box'] = b
        current['state_indices'].append(idx)
        current['detailed'] = min(current['reference_box'][2]-current['reference_box'][0],
                                  current['reference_box'][3]-current['reference_box'][1]) <= cfg.detailed_short_side_px
    return episodes


def _matches(e, r, window, detailed, cfg, revisit):
    anchor = r['anchor']
    if getattr(cfg, 'dynamic_ellipse', False):
        z=max(anchor['viewer_relative_zoom'],e['viewer_relative_zoom'])
        rx,ry=cfg.ellipse_width_px/(2*z),cfg.ellipse_height_px/(2*z)
        return ((e['x_px']-anchor['x_px'])/rx)**2+((e['y_px']-anchor['y_px'])/ry)**2 < 1.0
    if not detailed:
        # A later overview gaze within an already established detailed ROI
        # can revisit that ROI; it does not create an additional marker.
        if r['detail_reference_box'] is not None:
            return contains(r['detail_reference_box'], e)
        return local_distance(e, anchor, cfg) < 1.0
    ref = r['detail_reference_box']
    if ref is None:
        # Promotion needs a real historical gaze anchor inside this patch.
        return contains(window, anchor)
    if not contains(ref, e):
        return False
    if small_pan(ref, window, cfg, revisit):
        return True
    # A genuinely different zoomed subregion gets a new marker if the old
    # anchor is outside it. Enlargement/correction at the anchor is one ROI.
    nested = intersection(ref, window)/max(1e-12, min(area(ref), area(window)))
    return nested >= cfg.zoom_containment and contains(window, anchor)


def simplify(events, states, cfg=Config()):
    if min(cfg.overview_radius_x_px, cfg.overview_radius_y_px, cfg.detailed_short_side_px) <= 0:
        raise ValueError('Nonpositive spatial configuration')
    records = [dict(e) for e in sorted(events, key=lambda e:(e['start_time'], e['event_id']))]
    if len({e['event_id'] for e in records}) != len(records):
        raise ValueError('Duplicate event IDs')
    episodes = display_episodes(states, cfg)
    state_episode = {idx: p for p in episodes for idx in p['state_indices']}
    regions, visits = [], []
    active = None
    display_break = False
    for order, e in enumerate(records):
        p = state_episode[e['viewport_state_index']]
        window = p['reference_box']
        actual_window = box(states[e['viewport_state_index']])
        # A zoom correction can reveal tissue outside the smallest window.
        # At a detailed scale this is another candidate ROI, not overview
        # noise; a fixed reference still prevents chained pan accumulation.
        if p['detailed'] and contains(window, e):
            detailed = True
        elif min(actual_window[2]-actual_window[0],actual_window[3]-actual_window[1]) <= cfg.detailed_short_side_px:
            detailed, window = True, actual_window
        else:
            detailed = False
        if getattr(cfg, 'dynamic_ellipse', False):
            detailed = False
        e.update(source_order=order, region_id=None, visit_id=None, action_id=None,
                 display_episode_id=p['display_episode_id'], observation_mode='detailed_roi' if detailed else 'overview_search')
        if not e['input_valid']:
            if e['support_intervals']:
                raise ValueError('Excluded event cannot contribute gaze support')
            # A dropout on unchanged content does not by itself split a visit.
            if active and active['detail_reference_box'] is not None:
                ref = active['detail_reference_box']
                if intersection(ref, box(states[e['viewport_state_index']])) == 0:
                    display_break = True
            continue
        if not all(math.isfinite(e[k]) for k in ('x_px','y_px','start_time','end_time','viewer_relative_zoom')):
            raise ValueError('Nonfinite valid gaze event')
        candidate = None
        if active and not display_break and _matches(e, active, window, detailed, cfg, False):
            candidate = active
        if candidate is None:
            options = [r for r in regions if _matches(e, r, window, detailed, cfg, True)]
            if options:
                candidate = min(options, key=lambda r:math.hypot(e['x_px']-r['anchor']['x_px'],e['y_px']-r['anchor']['y_px']))
        if candidate is None:
            candidate = {'region_id':len(regions)+1, 'anchor':e,
                         'detail_reference_box':list(window) if detailed else None,
                         'source_event_ids':[], 'promoted_at_event_id':None}
            regions.append(candidate)
        elif detailed and candidate['detail_reference_box'] is None:
            candidate['detail_reference_box'] = list(window)
            candidate['promoted_at_event_id'] = e['event_id']
        region_id = candidate['region_id']
        revisiting = bool(candidate['source_event_ids'])
        if active is None or active['region_id'] != region_id or display_break:
            visits.append({'visit_id':len(visits)+1, 'region_id':region_id,
                           'events':[], 'revisit':revisiting})
        visit = visits[-1]
        visit['events'].append(e)
        candidate['source_event_ids'].append(e['event_id'])
        e.update(region_id=region_id,visit_id=visit['visit_id'])
        active, display_break = candidate, False

    steps = []
    for visit in visits:
        es = visit['events']
        # Include actual geometry changes through invalid gaze intervals, but
        # never create a node/visit solely from an operation without gaze.
        first, last = es[0]['viewport_state_index'], es[-1]['viewport_state_index']
        values = [states[i]['relative_zoom'] for i in range(first,last+1)]
        first_valid_at_state = next(e for e in records if e['input_valid'] and e['viewport_state_index']==first)
        if first > 0 and es[0]['event_id']==first_valid_at_state['event_id'] and not near_zoom(
                states[first-1]['relative_zoom'],states[first]['relative_zoom'],cfg.zoom_tolerance):
            # Include only the actual transition immediately before this
            # state's first supported visit, not the whole episode baseline.
            values = [states[first-1]['relative_zoom']] + values
        zs = []
        for z in values:
            if not zs or not near_zoom(zs[-1],z,cfg.zoom_tolerance):zs.append(float(z))
        step = {'step':len(steps)+1,'action_id':len(steps)+1,'visit_id':visit['visit_id'],
                'region_id':visit['region_id'],'revisit':visit['revisit'],
                'kind':'zoom_change' if len(zs)>1 else ('revisit' if visit['revisit'] else 'observation'),
                'start_time':es[0]['start_time'],'end_time':es[-1]['end_time'],'video_s':es[0]['video_s'],
                'zoom_sequence':zs,'final_zoom':es[-1]['viewer_relative_zoom'],
                'gaze_supported_zooms':sorted({e['viewer_relative_zoom'] for e in es}),
                'valid_support':interval_union([i for e in es for i in e['support_intervals']]),
                'source_event_ids':[e['event_id'] for e in es],
                'display_episode_ids':sorted({e['display_episode_id'] for e in es}),
                'source_order_first':es[0]['source_order'],'source_order_last':es[-1]['source_order'],
                'invalid_intervening_fragments':sum(not records[i]['input_valid'] for i in range(es[0]['source_order'],es[-1]['source_order']+1))}
        steps.append(step)
        for e in es:e['action_id']=step['action_id']
    rows = []
    for r in regions:
        a = r['anchor']
        rows.append({'region_id':r['region_id'],'anchor_event_id':a['event_id'],
                     **{k:a[k] for k in ('x_px','y_px','x_norm','y_norm')},
                     'detail_reference_box':r['detail_reference_box'],
                     'promoted_at_event_id':r['promoted_at_event_id'],
                     'source_event_ids':r['source_event_ids'],
                     'steps':[s['step'] for s in steps if s['region_id']==r['region_id']]})
    edges = []
    for a,b in zip(steps,steps[1:]):
        between=records[a['source_order_last']+1:b['source_order_first']]
        edges.append({'from_step':a['step'],'to_step':b['step'],
                      'from_region':a['region_id'],'to_region':b['region_id'],
                      'style':'dashed' if between else 'solid',
                      'intervening_fragments':len(between),
                      'meaning':'inferred chronological link, not a measured eye movement'})
    return {'config':asdict(cfg),'regions':rows,'steps':steps,'events':records,
            'display_episodes':episodes,'edges':edges}


def filter_observations(result, min_support=120.0):
    """Post-aggregation AND gate, applied separately to each original visit.

    Region membership and display geometry remain those of V4. Removed visits
    cannot be joined to sum their support. Audit data stays available; the
    trajectory contains only qualified visits and their supported regions.
    """
    if not math.isfinite(min_support) or min_support < 0:
        raise ValueError('min_support must be finite and nonnegative')
    out = deepcopy(result)
    out['config']['min_valid_support'] = float(min_support)
    records = out['events']
    byid = {e['event_id']: e for e in records}
    visits, steps, seen = [], [], set()
    for e in records:
        e.update(trajectory_retained=False, retained_step=None,
                 trajectory_exclusion_reason='input_quality_excluded' if not e['input_valid'] else None)
    for original in out['steps']:
        s = dict(original)
        es = [byid[i] for i in s['source_event_ids']]
        if not es or any(not e['input_valid'] for e in es):
            raise ValueError('Original visit lacks valid gaze evidence')
        # Union excludes gaps and prevents duplicated sample intervals from
        # counting twice; never use end_time-start_time as effective dwell.
        effective = interval_union([iv for e in es for iv in e['support_intervals']])
        s.update(original_step=original['step'], original_revisit=original['revisit'],
                 valid_support=effective, qualified=effective >= min_support,
                 exclusion_reason=None if effective >= min_support else 'below_min_valid_support',
                 step=None)
        if s['qualified']:
            s['step'] = len(steps)+1
            s['revisit'] = s['region_id'] in seen
            if s['kind'] != 'zoom_change':
                s['kind'] = 'revisit' if s['revisit'] else 'observation'
            seen.add(s['region_id'])
            steps.append(s)
        for e in es:
            e.update(trajectory_retained=s['qualified'],retained_step=s['step'],
                     trajectory_exclusion_reason=s['exclusion_reason'])
        visits.append(s)
    regions = []
    for region in out['regions']:
        retained = [s for s in steps if s['region_id']==region['region_id']]
        if not retained:
            continue
        r = dict(region)
        # The fixed grouping reference stays unchanged. The displayed marker
        # must be supported by a retained visit, not a rejected brief visit.
        r['grouping_anchor_event_id'] = r['anchor_event_id']
        r['grouping_anchor_x_px'], r['grouping_anchor_y_px'] = r['x_px'], r['y_px']
        anchor = byid[retained[0]['source_event_ids'][0]]
        r.update(anchor_event_id=anchor['event_id'],
                 **{k:anchor[k] for k in ('x_px','y_px','x_norm','y_norm')},
                 source_event_ids=[i for s in retained for i in s['source_event_ids']],
                 steps=[s['step'] for s in retained])
        regions.append(r)
    edges = []
    for a,b in zip(steps,steps[1:]):
        between = records[a['source_order_last']+1:b['source_order_first']]
        rejected_visits = [s for s in visits if not s['qualified'] and
                           a['original_step'] < s['original_step'] < b['original_step']]
        edges.append({'from_step':a['step'],'to_step':b['step'],
                      'from_region':a['region_id'],'to_region':b['region_id'],
                      'style':'dashed' if between or rejected_visits else 'solid',
                      'intervening_fragments':len(between),
                      'filtered_intervening_visits':len(rejected_visits),
                      'meaning':'inferred chronological link, not a measured eye movement'})
    out.update(regions=regions, steps=steps, edges=edges, visits_audit=visits,
               excluded_visits=[s for s in visits if not s['qualified']])
    return out

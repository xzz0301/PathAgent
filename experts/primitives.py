"""Exact interval-union and segment-distance helpers from the established workflow."""
import math

def interval_union(intervals):
    merged = []
    for a, b in sorted((float(a), float(b)) for a, b in intervals if b > a):
        if merged and a <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return sum(b-a for a, b in merged)

def segment_distance(p,a,b):
    vx,vy=b[0]-a[0],b[1]-a[1];den=vx*vx+vy*vy
    u=max(0,min(1,((p[0]-a[0])*vx+(p[1]-a[1])*vy)/den)) if den else 0
    return math.hypot(p[0]-a[0]-u*vx,p[1]-a[1]-u*vy)

from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio_ffmpeg


def overlay(frame: np.ndarray, fi: int, tracks: pd.DataFrame, config: dict) -> np.ndarray:
    im=cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
    oc=config["output"]
    for ident,g in tracks.groupby("track_id"):
        hist=g[(g.frame<=fi)&(g.frame>=fi-oc["history"])].sort_values("frame")
        previous=None
        for row in hist.itertuples():
            if row.observed:
                p=(int(round(row.center_x)),int(round(row.center_y)))
                if previous is not None and previous[0]==row.frame-1 and row.status!="manual_anchor":
                    cv2.line(im,previous[1],p,(0,255,255),oc["trajectory_thickness"],cv2.LINE_AA)
                previous=(row.frame,p)
            else: previous=None
        current=g[g.frame==fi]
        if current.empty: continue
        row=current.iloc[0]
        if row.observed:
            p=(int(round(row.center_x)),int(round(row.center_y)))
            cv2.circle(im,p,int(round(row.radius_px)),(0,220,0),oc["circle_thickness"],cv2.LINE_AA)
            cv2.circle(im,p,2,(0,255,255),-1)
            if oc["show_ids"]:
                tx=max(2,min(im.shape[1]-35,p[0]+int(row.radius_px)+3));ty=max(16,min(im.shape[0]-4,p[1]))
                cv2.putText(im,str(ident),(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),3,cv2.LINE_AA)
                cv2.putText(im,str(ident),(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1,cv2.LINE_AA)
            if oc["show_bbox"]:
                cv2.rectangle(im,(int(row.x1),int(row.y1)),(int(row.x2),int(row.y2)),(0,220,0),1)
    footer=np.zeros((40,im.shape[1],3),np.uint8)
    now=tracks[tracks.frame==fi]
    lost=now.loc[~now.observed,"track_id"].tolist()
    cv2.putText(footer,f"Frame {fi} | selected {len(now)} | observed {int(now.observed.sum())}",(8,15),cv2.FONT_HERSHEY_SIMPLEX,.42,(220,220,220),1,cv2.LINE_AA)
    label="Needs review: "+", ".join(map(str,lost)) if lost else "Yellow: observed center path | green: selected ROI"
    cv2.putText(footer,label,(8,32),cv2.FONT_HERSHEY_SIMPLEX,.40,(0,200,255) if lost else (160,210,160),1,cv2.LINE_AA)
    return np.vstack([im,footer])


def render_video(frames: list[np.ndarray],tracks: pd.DataFrame,config: dict,fps: float,output: Path) -> None:
    h,w=frames[0].shape
    writer=imageio_ffmpeg.write_frames(str(output),(w,h+40),fps=fps,codec="libx264",pix_fmt_in="rgb24",pix_fmt_out="yuv420p",quality=8,
                                      macro_block_size=1,output_params=["-movflags","+faststart"])
    writer.send(None)
    try:
        for i,f in enumerate(frames):
            im=overlay(f,i,tracks,config)
            writer.send(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    finally: writer.close()
    cap=cv2.VideoCapture(str(output));decoded=0
    while cap.read()[0]: decoded+=1
    cap.release()
    if decoded!=len(frames): raise RuntimeError(f"Output video verification failed: {decoded}/{len(frames)}")


def trajectory_segments(group: pd.DataFrame):
    """Keep measured anchors, but do not draw a displacement into a manual correction."""
    g=group.sort_values("frame")
    breaks=(~g.observed | ~g.observed.shift(1,fill_value=False) |
            g.frame.diff().ne(1) | g.status.eq("manual_anchor"))
    for _, segment in g.groupby(breaks.cumsum()):
        observed=segment[segment.observed]
        if len(observed): yield observed


def plots(frames: list[np.ndarray],tracks: pd.DataFrame,output: Path,config: dict) -> None:
    output.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(7,7),dpi=160)
    ax.imshow(frames[0],cmap="gray",vmin=0,vmax=255)
    for ident,g in tracks.groupby("track_id"):
        g=g.sort_values("frame")
        for segment in trajectory_segments(g):
            ax.plot(segment.center_x,segment.center_y,color="#ffdd00",lw=1.2)
        obs=g[g.observed]
        if len(obs): ax.text(obs.center_x.iloc[0],obs.center_y.iloc[0],str(ident),color="#39ff14",fontsize=10)
    ax.set(xlabel="x (pixel)",ylabel="y (pixel)",title="Selected particle trajectories - raw image coordinates")
    fig.tight_layout();fig.savefig(output/"trajectories.png");plt.close(fig)
    directory=output/"particle_trajectories";directory.mkdir(exist_ok=True)
    for ident,g in tracks.groupby("track_id"):
        g=g.sort_values("frame")
        fig,axes=plt.subplots(1,2,figsize=(9,4),dpi=150)
        for segment in trajectory_segments(g):
            axes[0].plot(segment.center_x,segment.center_y,".-",color="#177f51",markersize=3)
            axes[1].plot(segment.frame,segment.center_x,color="tab:blue")
            axes[1].plot(segment.frame,segment.center_y,color="tab:orange")
        axes[0].invert_yaxis();axes[0].set_aspect("equal",adjustable="datalim")
        axes[0].set(title=f"Particle {ident}: image coordinates",xlabel="x (px)",ylabel="y (px)")
        axes[1].plot([],[],color="tab:blue",label="x raw")
        axes[1].plot([],[],color="tab:orange",label="y raw")
        axes[1].set(title="Gaps and manual corrections break the line",xlabel="Frame",ylabel="Coordinate (px)")
        axes[1].legend();fig.tight_layout();fig.savefig(directory/f"particle_{ident:03d}.png");plt.close(fig)
    chosen=np.linspace(0,len(frames)-1,min(6,len(frames)),dtype=int)
    panels=[cv2.resize(overlay(frames[i],i,tracks,config),(360,386)) for i in chosen]
    while len(panels)<6:panels.append(np.zeros_like(panels[0]))
    cv2.imwrite(str(output/"tracking_contact_sheet.jpg"),np.vstack([np.hstack(panels[:3]),np.hstack(panels[3:])]))
    cv2.imwrite(str(output/"first_frame_overlay.png"),overlay(frames[0],0,tracks,config))

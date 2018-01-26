import skvideo.io
import skvideo.datasets
import numpy as np
from os import path
from moviepy.editor import *
from PIL import Image, ImageDraw, ImageFont



# PURPOSE
# =======
# The goal of this test file is to experiment with video annotation functions of moviepy, skvideo
# The task is to add a lateral band to the video and fill it with text

"""
#### SKVIDEO
# READ VIDEO
# ==========
filepath    =   path.join('/home/younesz/Downloads/test_video.mp4')
# Read frame shape and header
videoclip   =   skvideo.io.vreader(filepath)
videoinfo   =   skvideo.io.ffprobe(filepath)
w,h,r,nf    =   int(videoinfo['video']['@width']), int(videoinfo['video']['@height']), int(eval(videoinfo['video']['@r_frame_rate'])), int(videoinfo['video']['@nb_frames'])
# Append black lateral band: 100 pixels wide
videoclip2  =   np.zeros([nf, h, w+100, 3])
count       =   0
for frame in videoclip:
    frame2  =   np.concatenate( (frame, np.zeros([h, 100, 3])), axis=1)
    videoclip2[count,:,:,:]     =   frame2
    count   +=  1
# Write video
writer      =   skvideo.io.FFmpegWriter('/home/younesz/Downloads/test_output.mp4')
for ifr in range(nf):
    writer.writeFrame(videoclip2[ifr,:,:,:])
writer.close()
"""

def read_gametape(videopath):
    # Read frame shape and header
    videoclip       =   VideoFileClip(gamepath)
    [w, h], r, d    =   videoclip.size, videoclip.fps, videoclip.duration
    nf              =   int(r * d)
    return w,h,r,d,nf

def parametrize_feed(w):
    # Paramterize display for any video
    txt_param   = {'tit_size': 10}
    away_param  = {'tit_pos': (10, 10), 'grph_pos': (10, 30)}
    home_param  = {'tit_pos': (w - 70, 10), 'grph_pos': (w - 70, 30)}
    gen_param   = {'grph_opac': 0.5, 'grph_width': 50, 'grph_height': 65, 'grph_xbnds': np.array([-0.31, 1.37]),
                    'grph_ybnds': np.array([-0.07, 0.99]),
                    'pl_adjust': [-txt_param['tit_size'] / 4, -txt_param['tit_size'] * 1.5]}
    return {'text':txt_param, 'away':away_param, 'home':home_param, 'general':gen_param}


def translate_pred2coord(pred, p):
    coord_rel   =   np.divide( np.abs( np.subtract( pred, [p['grph_xbnds'][0], p['grph_ybnds'][1]] ) ),
                    np.concatenate([np.diff(p['grph_xbnds']), np.diff(p['grph_ybnds'])]) )
    coord_abs   =   np.multiply( coord_rel, [p['grph_width'], p['grph_height']] )
    return list(coord_abs)


def make_feed()



#### MOVIEPY
# READ VIDEO
# ==========
season      =   '20142015'
gcode       =   '20020'
gamepath    =   'gameId_'+season+'0'+gcode+'.mp4'
gamepath    =   '/home/younesz/Downloads/test_video.mp4'
datapath    =   '/home/younesz/Documents/Code/Python/ReinforcementLearning/NHL/playbyplay/data'

# AWAY TEAM INFO
# ==============
# Text clip:  home team title
txt_clip_a_t=   ( TextClip("Away team:",fontsize=txt_param['tit_size'],color='white',stroke_width=2)
             .set_position(away_param['tit_pos'], relative=False)
             .set_duration(d)
             .set_start(0) )
# Graph mask : home team
grph_clip_a =   ( ImageClip('/home/younesz/Downloads/mask.png')
                  .set_position(away_param['grph_pos'], relative=False)
                  .set_opacity(gen_param['grph_opac'])
                  .set_duration(d)
                  .set_start(0)
                  .fx( vfx.resize, width=gen_param['grph_width'], height=gen_param['grph_height']) )
# Display one player of the home team: prediction=[1,0]
pl_a_1      =   ( TextClip(".",fontsize=2*txt_param['tit_size'],color='orange',stroke_width=2)
                .set_position( tuple( (np.array(away_param['grph_pos']) + gen_param['pl_adjust'] + translate_pred2coord([1,0], gen_param)).astype('int') ), relative=False)
                .set_duration(d+2)
                .set_start(1) )

# HOME TEAM INFO
# ==============
# Text clip:  home team title
txt_clip_h_t =   ( TextClip("Home team:",fontsize=txt_param['tit_size'],color='white',stroke_width=2)
                .set_position(home_param['tit_pos'], relative=False)
                .set_duration(d)
                .set_start(0) )
# Graph mask : home team
grph_clip_h   =   ( ImageClip('/home/younesz/Downloads/mask.png')
                .set_position(home_param['grph_pos'], relative=False)
                .set_opacity(gen_param['grph_opac'])
                .set_duration(d)
                .set_start(0)
                .fx( vfx.resize, width=gen_param['grph_width'], height=gen_param['grph_height']) )
# Display one player of the home team: prediction=[0,0]
pl_h_1      =   ( TextClip(".",fontsize=2*txt_param['tit_size'],color='green',stroke_width=2)
                .set_position( tuple( (np.array(home_param['grph_pos']) + gen_param['pl_adjust'] + translate_pred2coord([0,0.8], gen_param)).astype('int') ), relative=False)
                .set_duration(d+2)
                .set_start(1) )




#def make_pbp_clip():



# Superimpose videos
result      =   CompositeVideoClip([videoclip, txt_clip_h_t, grph_clip_h, txt_clip_a_t, grph_clip_a, pl_a_1, pl_h_1]) # Overlay text on video
result.write_videofile('/home/younesz/Downloads/test_output.mp4',fps=r)


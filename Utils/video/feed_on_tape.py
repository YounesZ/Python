import pickle
import numpy as np
from os import path
from copy import deepcopy
from moviepy.editor import *


# PURPOSE
# =======
# The goal of this module is to produce a live feed of information on top of a game tape


def decode_state(state, dims=[3,5,10]):
    dstate  =   []
    wrkD    =   deepcopy(dims)
    while len(wrkD)>1:
        dimii   =   wrkD.pop(0)
        dstate.append( int(np.fix(state/np.prod(wrkD))) )
        state   -=  dstate[-1] * np.prod(wrkD)
    dstate.append( state )
    return tuple(dstate)


def read_feeddata(datapath, season, gcode):
    # Read data
    GAME_data   =   pickle.load( open(path.join(datapath, 'GAME_data.p'), 'rb') )
    PLAYER_data =   pickle.load( open(path.join(datapath, 'PLAYER_data.p'), 'rb') )
    RL_data     =   pickle.load( open(path.join(datapath, 'RL_teaching_data.p'), 'rb') )
    nAct        =   RL_data['nActions']
    nSt         =   RL_data['nStates']
    RL_data     =   RL_data['RL_data']
    # Pick season and game
    GAME_data   =   GAME_data[GAME_data['gameCode'] == int(gcode)]
    GAME_data   =   GAME_data[GAME_data['season'] == int(season)]
    PLAYER_data =   PLAYER_data[PLAYER_data['gameCode'] == int(gcode)]
    PLAYER_data =   PLAYER_data[PLAYER_data['season'] == int(season)]
    #Qvalues    =   pickle.load( open(path.join(datapath, 'Q_values.p')) )
    Qvalues     =   np.random.random([3,5,10,nAct]) - 0.5
    return GAME_data, PLAYER_data, RL_data, Qvalues


def read_gametape(videopath):
    # Read frame shape and header
    videoclip       =   VideoFileClip(videopath)
    [w, h], r, d    =   videoclip.size, videoclip.fps, videoclip.duration
    nf              =   int(r * d)
    return videoclip, w,h,r,d,nf


def parametrize_feed(w):
    # Paramterize display for any video
    txt_param   = {'tit_size': 10, 'plName_size':4}
    away_param  = {'tit_pos': (10, 10), 'grph_pos': (10, 30)}
    home_param  = {'tit_pos': (w - 70, 10), 'grph_pos': (w - 70, 30)}
    gen_param   = {'grph_opac': 0.5, 'grph_width': 50, 'grph_height': 65, 'grph_xbnds': np.array([-0.31, 1.37]),
                    'grph_ybnds': np.array([-0.07, 0.99]),
                    'pl_adjust': [-txt_param['tit_size'] / 4, -txt_param['tit_size'] * 1.5]}
    return {'text':txt_param, 'away':away_param, 'home':home_param, 'general':gen_param}


def translate_pred2coord(pred, p={}):
    coord_rel   =   np.divide( np.abs( np.subtract( pred, [p['grph_xbnds'][0], p['grph_ybnds'][1]] ) ),
                    np.concatenate([np.diff(p['grph_xbnds']), np.diff(p['grph_ybnds'])]) )
    coord_abs   =   np.multiply( coord_rel, [p['grph_width'], p['grph_height']] )
    return list(coord_abs)


def make_feed(GAME_data, PLAYER_data, RL_data, Qvalues, P):
    # First the static part
    # =====================
    clips_static    =   make_feed_static(GAME_data, P)

    # Second the dynamic part
    # =======================
    nRows           =   len(GAME_data)
    nRows           =   10
    Qrange          =   [np.min(Qvalues), np.max(Qvalues)]
    clips_dynamic   =   []
    for iR  in range(nRows):
        [playersID_a, playersID_h], start, dur, equalS, diff, per  =   GAME_data.iloc[iR][['playersID', 'onice', 'iceduration', 'equalstrength', 'differential', 'period']]
        Qvalue_iR       =   Qvalues[decode_state(RL_data.iloc[iR]['state'])][RL_data.iloc[iR]['action']]
        clips_dynamic   +=  make_feed_dynamic(PLAYER_data.loc[playersID_a], PLAYER_data.loc[playersID_h], start, dur, equalS, diff, Qvalue_iR, P)[0]
    return clips_static + clips_dynamic


def make_feed_static(GAME_data, P):
    # AWAY TEAM:
    # ==========
    # Text clip:  home team title
    txt_clip_a_t = (TextClip("Away team:", fontsize=P['text']['tit_size'], color='white', stroke_width=2)
                    .set_position(P['away']['tit_pos'], relative=False)
                    .set_duration(d)
                    .set_start(0))
    # Graph mask : home team
    grph_clip_a = (ImageClip('/home/younesz/Downloads/mask.png')
                   .set_position(P['away']['grph_pos'], relative=False)
                   .set_opacity(P['general']['grph_opac'])
                   .set_duration(d)
                   .set_start(0)
                   .fx(vfx.resize, width=P['general']['grph_width'], height=P['general']['grph_height']))

    # HOME TEAM:
    # ==========
    # Text clip:  home team title
    txt_clip_h_t = (TextClip("Home team:", fontsize=P['text']['tit_size'], color='white', stroke_width=2)
                    .set_position(P['home']['tit_pos'], relative=False)
                    .set_duration(d)
                    .set_start(0))
    # Graph mask : home team
    grph_clip_h = (ImageClip('/home/younesz/Downloads/mask.png')
                   .set_position(P['home']['grph_pos'], relative=False)
                   .set_opacity(P['general']['grph_opac'])
                   .set_duration(d)
                   .set_start(0)
                   .fx(vfx.resize, width=P['general']['grph_width'], height=P['general']['grph_height']))
    return [txt_clip_a_t, grph_clip_a, txt_clip_h_t, grph_clip_h]


def make_feed_dynamic(players_a, players_h, start, dur, equalS, diff, Qv, P):
    # Pre-settings
    colChx  =   {0:'black', 1:'red', 2:'blue'}
    algnChx =   {0:'North', 1:'South', 2:'center'}
    # AWAY TEAM:
    # ==========
    # Players projection on graph
    p_proj  =   [translate_pred2coord(x, P['general']) for x in list( players_a[['pred_ross', 'pred_selke']].values )]
    pl_a_pt = [(TextClip(".", fontsize=2 * P['text']['tit_size'], color=colChx[y], stroke_width=2)
              .set_position(tuple((np.array(P['away']['grph_pos']) + P['general']['pl_adjust'] + x).astype('int')), relative=False)
              .set_duration(dur)
              .set_start(start)) for x,y in zip(p_proj, players_a['class'])]
    # Players names on top of projection
    pl_a_nm = [(TextClip(x.split(" ")[-1], fontsize=2 * P['text']['plName_size'], color=colChx[y], align=algnChx[y], stroke_width=2)
              .set_position(tuple((np.array(P['away']['grph_pos']) + P['general']['pl_adjust'] + z).astype('int')), relative=False)
              .set_duration(dur)
              .set_start(start)) for x,y,z in zip(players_a['firstlast'].values, players_a['class'], p_proj)]

    # HOME TEAM:
    # ==========
    # Players projection on graph
    p_proj  = [translate_pred2coord(x, P['general']) for x in list(players_h[['pred_ross', 'pred_selke']].values)]
    pl_h_pt = [(TextClip(".", fontsize=2 * P['text']['tit_size'], color=colChx[y], stroke_width=2)
               .set_position(tuple((np.array(P['home']['grph_pos']) + P['general']['pl_adjust'] + translate_pred2coord(x, P['general'])).astype('int')), relative=False)
               .set_duration(dur)
               .set_start(start)) for x, y in zip(list(players_h[['pred_ross', 'pred_selke']].values), players_h['class'])]
    # Players names on top of projection
    pl_h_nm = [(TextClip(x.split(" ")[-1], fontsize=2 * P['text']['plName_size'], color=colChx[y], align=algnChx[y], stroke_width=2)
                .set_position(tuple((np.array(P['home']['grph_pos']) + z).astype('int')), relative=False)
                .set_duration(dur)
                .set_start(start)) for x,y,z in zip(players_h['firstlast'].values, players_h['class'],p_proj)]

    # GAME INFO:
    # ==========

    print('Working on that part...')
    return  [pl_a_pt+pl_a_nm+pl_h_pt+pl_h_nm]


# LAUNCHER
# ========
season      =   '20142015'
gcode       =   '20020'
videopath   =   'gameId_'+season+'0'+gcode+'.mp4'
videopath   =   '/home/younesz/Downloads/test_video.mp4'
datapath    =   '/home/younesz/Documents/Code/Python/ReinforcementLearning/NHL/playbyplay/data'


# Prepare video
VID, w,h,r,d,nf =   read_gametape(videopath)
P               =   parametrize_feed(w)

# Read feed data
GAME_data, PLAYER_data, RL_data, Qvalues    =   read_feeddata(datapath, season, gcode)
feed_clips      =   make_feed(GAME_data, PLAYER_data, RL_data, Qvalues, P)

# Superimpose videos
result          =   CompositeVideoClip([VID]+feed_clips) # Overlay text on video
result.write_videofile('/home/younesz/Downloads/test_output.mp4',fps=r)



"""


#def make_pbp_clip():





"""
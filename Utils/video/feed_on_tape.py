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
    Qvalues    =   pickle.load( open(path.join(datapath, 'RL_action_values.p'), 'rb') )['action_values']
    #Qvalues     =   np.random.random([3,5,10,nAct]) - 0.5
    return GAME_data, PLAYER_data, RL_data, Qvalues


def read_gametape(videopath):
    # Read frame shape and header
    videoclip       =   VideoFileClip(videopath)
    [w, h], r, d    =   videoclip.size, videoclip.fps, videoclip.duration
    nf              =   int(r * d)
    return videoclip, w,h,r,d,nf


def parametrize_feed(w,h):
    # Paramterize display for any video
    txt_param   =   {'tit_size': 30, 'plName_size':20, 'info_pos':(0.37, 0.13)}
    away_param  =   {'tit_pos': (10, 20), 'grph_pos': (10, 60)}
    home_param  =   {'tit_pos': (w - 170, 20), 'grph_pos': (w - 170, 60)}
    gen_param   =   {'grph_opac': 0.5, 'grph_size': [150, 195], 'grph_xbnds': np.array([-0.31, 1.37]),
                    'grph_ybnds': np.array([-0.07, 0.99]), 'clb_pos':[(w-300)/2, h-50], 'clb_sz':[300, 300*89/555],
                    'pl_adjust': [-3,-40], 'pnt_sz':50}
    return {'text':txt_param, 'away':away_param, 'home':home_param, 'general':gen_param}


def make_colorbar():
    import matplotlib.pyplot as plt
    FIG     =   plt.figure()
    CLB     =   plt.imshow(np.tile(range(1, 61), [4, 1]))
    CLB.set_cmap('coolwarm')
    plt.gca().set_xticks([0, 30, 59])
    plt.gca().set_xticklabels(['low', 'neutral', 'high'], fontdict={'fontsize':20})
    plt.gca().set_yticks([])
    plt.savefig('/home/younesz/Downloads/clb_t.png', transparent=True)
    plt.savefig('/home/younesz/Downloads/clb.png', transparent=False)
    plt.close(FIG)


def translate_pred2coord(pred, p={}):
    coord_rel   =   np.divide( np.abs( np.subtract( pred, [p['grph_xbnds'][0], p['grph_ybnds'][1]] ) ),
                    np.concatenate([np.diff(p['grph_xbnds']), np.diff(p['grph_ybnds'])]) )
    coord_abs   =   np.multiply( coord_rel, [p['grph_size'][0], p['grph_size'][1]] )
    return list(coord_abs)


def make_feed(GAME_data, PLAYER_data, RL_data, Qvalues, P):
    # First the static part
    # =====================
    clips_static    =   make_feed_static(GAME_data, P)

    # Second the dynamic part
    # =======================
    nRows           =   len(GAME_data)
    #nRows           =   5
    Qrange          =   [np.min(Qvalues), np.max(Qvalues)]
    clips_dynamic   =   []
    for iR  in range(nRows):
        [playersID_h, playersID_a], start, dur, equalS, diff, per, Hteam, Ateam =   GAME_data.iloc[iR][['playersID', 'onice', 'iceduration', 'equalstrength', 'differential', 'period', 'hometeam', 'awayteam']]
        Qvalue_iR       =   (Qvalues[decode_state(RL_data.iloc[iR]['state'])][RL_data.iloc[iR]['action']] - Qrange[0]) / np.diff(Qrange)
        clips_dynamic   +=  make_feed_dynamic(PLAYER_data.loc[playersID_a], PLAYER_data.loc[playersID_h], start, dur, equalS, diff, per, Hteam, Ateam, Qvalue_iR, P)[0]
    return clips_static + clips_dynamic


def make_feed_static(GAME_data, P):
    # AWAY TEAM:
    # ==========
    # Text clip:  home team title
    txt_clip_a_t = (TextClip("away:", fontsize=P['text']['tit_size']-5, color='white', stroke_width=2)
                    .set_position(P['away']['tit_pos'], relative=False)
                    .set_duration(d)
                    .set_start(0))
    # Graph mask : home team
    grph_clip_a = (ImageClip('/home/younesz/Downloads/mask.png')
                   .set_position(P['away']['grph_pos'], relative=False)
                   .set_opacity(P['general']['grph_opac'])
                   .set_duration(d)
                   .set_start(0)
                   .fx(vfx.resize, newsize=P['general']['grph_size']))

    # HOME TEAM:
    # ==========
    # Text clip:  home team title
    txt_clip_h_t = (TextClip("home:", fontsize=P['text']['tit_size']-5, color='white', stroke_width=2)
                    .set_position(P['home']['tit_pos'], relative=False)
                    .set_duration(d)
                    .set_start(0))
    # Graph mask : home team
    grph_clip_h = (ImageClip('/home/younesz/Downloads/mask.png')
                   .set_position(P['home']['grph_pos'], relative=False)
                   .set_opacity(P['general']['grph_opac'])
                   .set_duration(d)
                   .set_start(0)
                   .fx(vfx.resize, newsize=P['general']['grph_size']))

    # COLOR BAR:
    # ==========
    # Q-values
    qval_clb    = (ImageClip('/home/younesz/Downloads/clb_t.png')
                   .set_position(P['general']['clb_pos'], relative=False) #.set_opacity(P['general']['grph_opac'])
                   .set_duration(d)
                   .set_start(0)
                   .fx(vfx.resize, width=P['general']['clb_sz'][0]) )
                   #.fx(vfx.resize, width=P['general']['clb_sz'][0], height=P['general']['clb_sz'][1]))

    return [txt_clip_a_t, grph_clip_a, txt_clip_h_t, grph_clip_h, qval_clb]


def make_feed_dynamic(players_a, players_h, start, dur, equalS, diff, per, Hteam, Ateam, Qv, P):
    # Pre-settings
    colChx  =   {0:'black', 1:'red', 2:'blue'}
    algnChx =   {0:'North', 1:'South', 2:'center'}

    # AWAY TEAM:
    # ==========
    # Players projection on graph
    p_proj  =   [translate_pred2coord(x, P['general']) for x in list( players_a[['pred_ross', 'pred_selke']].values )]
    i_sort  =   - np.argsort( np.array(p_proj)[:,-1] ) + 1
    #p_proj  =   [translate_pred2coord(x, P['general']) for x in [[0,1], [0,0], [1,0]]]
    pl_a_pt =   [(TextClip(".", fontsize=P['general']['pnt_sz'], color=colChx[y], stroke_width=2)
                .set_position(tuple((np.array(P['away']['grph_pos']) + P['general']['pl_adjust'] + x).astype('int')), relative=False)
                .set_duration(dur)
                .set_start(start)) for x,y in zip(p_proj, players_a['class'])]
    # Players names on top of projection
    pl_a_nm =   [(TextClip(x.split(" ")[-1], fontsize=P['text']['plName_size'], color=colChx[y], align=algnChx[y], stroke_width=2)
                .set_position(tuple((np.array(P['away']['grph_pos']) + z + np.array([10, -w*10])).astype('int')), relative=False)
                .set_duration(dur)
                .set_start(start)) for w,x,y,z in zip(i_sort, players_a['firstlast'].values, players_a['class'], p_proj)]
    # Team name
    tm_a_nm =   [(TextClip(Ateam, fontsize=P['text']['tit_size']-5, color='white', stroke_width=2)
                .set_position( tuple( np.add(P['away']['tit_pos'],[100, 0]) ), relative=False)
                .set_duration(d)
                .set_start(0))]

    # HOME TEAM:
    # ==========
    # Players projection on graph
    p_proj  =   [translate_pred2coord(x, P['general']) for x in list(players_h[['pred_ross', 'pred_selke']].values)]
    i_sort  =   - np.argsort(np.array(p_proj)[:, -1]) + 1
    pl_h_pt =   [(TextClip(".", fontsize=P['general']['pnt_sz'], color=colChx[y], stroke_width=2)
                .set_position(tuple((np.array(P['home']['grph_pos']) + P['general']['pl_adjust'] + x).astype('int')), relative=False)
                .set_duration(dur)
                .set_start(start)) for x, y in zip(p_proj, players_h['class'])]
    # Players names on top of projection
    pl_h_nm =   [(TextClip(x.split(" ")[-1], fontsize=P['text']['plName_size'], color=colChx[y], align=algnChx[y], stroke_width=2)
                .set_position(tuple((np.array(P['home']['grph_pos']) + z + np.array([10, -w*10])).astype('int')), relative=False)
                .set_duration(dur)
                .set_start(start)) for w,x,y,z in zip(i_sort, players_h['firstlast'].values, players_h['class'], p_proj)]
    # Team name
    tm_h_nm =   [(TextClip(Hteam, fontsize=P['text']['tit_size']-5, color='white', stroke_width=2)
                .set_position( tuple( np.add(P['home']['tit_pos'],[100,0]) ), relative=False)
                .set_duration(d)
                .set_start(0))]

    # GAME INFO:
    # ==========
    # Period, differential
    gm_pd   =   [(TextClip('Period: %i, differential: %i' %(per, diff), fontsize=P['text']['tit_size']-10, color='white', stroke_width=2, method='label', align='center')
                .set_position(P['text']['info_pos'], relative=True)
                .set_duration(dur)
                .set_start(start))]

    # Q-VALUE:
    # ========
    # Set bar
    qval    =   [(TextClip('|', fontsize=P['text']['tit_size']+10, color='black', stroke_width=3, method='label', align='center')
                .set_position( np.add(np.add(P['general']['clb_pos'],[Qv*(P['general']['clb_sz'][0]-36)+17, P['general']['clb_sz'][1]/1.75]), P['general']['pl_adjust']), relative=False)
                .set_duration(dur)
                .set_start(start))]

    return  [pl_a_pt+pl_a_nm+tm_a_nm+pl_h_pt+pl_h_nm+tm_h_nm+gm_pd+qval]


# LAUNCHER
# ========
season      =   '20132014'
gcode       =   '20727'
videopath   =   '/home/younesz/Downloads/gameId_'+season[:4]+'0'+gcode+'.mp4'
#videopath   =   '/home/younesz/Downloads/test_hockey_120s.mp4'
datapath    =   '/home/younesz/Documents/Code/Python/ReinforcementLearning/NHL/playbyplay/data'


# Prepare video
VID, w,h,r,d,nf =   read_gametape(videopath)
P               =   parametrize_feed(w,h)

# Read feed data
GAME_data, PLAYER_data, RL_data, Qvalues    =   read_feeddata(datapath, season, gcode)
feed_clips      =   make_feed(GAME_data, PLAYER_data, RL_data, Qvalues, P)

# Superimpose videos
result          =   CompositeVideoClip([VID]+feed_clips) # Overlay text on video
result.write_videofile('/home/younesz/Downloads/gameId_'+season[:4]+'0'+gcode+'_wFeed.mp4',fps=r)



"""


#def make_pbp_clip():





"""
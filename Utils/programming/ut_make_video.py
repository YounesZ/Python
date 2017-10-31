def make_video(images, outvid, fps=5, size=None, is_color=True, format="XVID"):

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # Call writer
    Writer  =   animation.writers['ffmpeg']
    writer  =   Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)

    # Create figure
    fig2    =   plt.figure()

    x = np.arange(-9, 10)
    y = np.arange(-9, 10).reshape(-1, 1)
    base = np.hypot(x, y)
    ims = []
    for add in np.arange(15):
        ims.append((plt.pcolor(x, y, base + add, norm=plt.Normalize(0, 30)),))


    # Create animation
    im_ani  =   animation.ArtistAnimation(fig2, ims, interval=50, blit=True)
    im_ani.save(outvid, writer=writer)


"""
    import skvideo.io
    import numpy as np

    # Initiate the writer
    nImages     =   len(images)
    imShape     =   images[0].shape
    writer      =   skvideo.io.FFmpegWriter(outvid, inputdict={'-r':str(fps)+'/1'},
                                            outputdict={'-vcodec': 'libx264', '-b': '300000000', '-r':str(fps)+'/1'})

    # Write frames
    for image in images:
        image   =   (image * 255).astype(np.uint8)
        writer.writeFrame(image)
    writer.close()
"""

"""
    from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
    import os
    fourcc  =   VideoWriter_fourcc(*format)
    vid     =   None
    for image in images:

        # Check for image type
        if type(image) is str:  # Otherwise assume it's a matrix
            if not os.path.exists(image):
                raise FileNotFoundError(image)
            img     =   imread(image)
        else:
            img     =   image
        if vid is None:
            if size is None:
                size=   img.shape[1], img.shape[0]
            vid     =   VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] !=   img.shape[1] and size[1] != img.shape[0]:
            img     =   resize(img, size)
        vid.write(img)
    vid.release()
    return vid
"""


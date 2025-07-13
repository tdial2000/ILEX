##===============================================##
##===============================================##
## Author: Tyson Dial
## Email: tdial@swin.edu.au
## Last Updated: 04/06/2025 
##
##
## 
## 
## Library of interactive functions for ilex 
## plotting
##
##===============================================##
##===============================================##
# imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import time
from ilex.data import *
import warnings

RAPIDKEYTHRESHOLD = 0.6



class _rect:

    def __init__(self, fig, ax, func = None, color = 'b'):

        # ax to draw rectangle on
        # func, function to evaluate once rectangle has been drawn

        self.fig = fig
        self.ax = ax
        self.func = func
        self.is_button_held = False
        self.color = color

        self.button_press_event = self.fig.canvas.mpl_connect("button_press_event", self._on_button_press)
        self.button_move_event = self.fig.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.button_release_event = self.fig.canvas.mpl_connect("button_release_event", self._on_button_release)

    
    def _on_button_press(self, event):

        print("press here")

        # create rectangle and start taking note of x and y coordinates

        if (event.button == 1) and self.ax.in_axes(event):
            self.is_button_held = True
        else:
            return

        self.px = event.xdata
        self.py = event.ydata

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()


        self.rect = self.ax.fill_between([self.px, self.px], self.py, self.py, color = self.color,
                                            alpha = 0.2)

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

        return
    

    def _on_move(self, event):
        print("here")
        if self.is_button_held and self.ax.in_axes(event):

            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()

            self.rect.remove()
            self.rect = self.ax.fill_between([self.px, event.xdata], self.py, event.ydata, color = self.color,
                                                alpha = 0.2)

            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)

            self.fig.canvas.draw()


        return
    

    def _on_button_release(self, event):

        if (event.button == 1) and self.is_button_held:
            self.is_button_held = False

            self.nx = event.xdata
            self.ny = event.ydata

            # create rect event instance

            revent = rect_event(self.px, self.nx, self.py, self.ny, self.ax)

            # remove rect
            self.rect.remove()

            # run func
            if self.func is None:
                print("There is no function to evaluate with rect event")
            else:
                self.func(revent)

            # update ax
            self.fig.canvas.draw()

        return
    

    def rect_disconnect(self):

        self.fig.canvas.mpl_disconnect(self.button_press_event)
        self.fig.canvas.mpl_disconnect(self.button_move_event)
        self.fig.canvas.mpl_disconnect(self.button_release_event)

        return




class _MouseZapper:
    """
    Class for zapping channels using the mouse.

    press "z": Activate zapping mode
        LM: zap single channels
        RM: define lower and upper bounds of freq region to zap
            press "shift": zap region
        press "tab": revert previous zapping
    """

    def __init__(self, fig, ax, ds, freqs, zapchan = None, xlim = None):

        self.print_info()

        if xlim is None:
            xlim = [0.0, 1.0]

        self.xlim = xlim

        self.fig = fig
        self.ax = ax
        self.ds = ds
        self.freqs = freqs
        self.df = np.abs(self.freqs[0] - self.freqs[1])

        # copy of ds for zapping purposes
        self.zap_ds = None
        self.zaps = []
        self.base_zapchan = ""
        self.zapchan = None
        self.final_zapchan = zapchan
        if zapchan is not None:
            self.base_zapchan = zapchan

        # flags
        self.zapmode = False
        self.mouse_click_event = None

        self.keys_pressed = set()

        self.last_key_event = None
        self.last_key_press = time.time()

        self._list_of_auto_alg = ["median", "abs"]
        self._auto_alg = "median"


        # artists
        self._artists = {'ds': None, 'zapatch': None, 
                         'LMline': None, "RMlines": []}
        self._last_artist = None
                         
        self.key_press_event = self.fig.canvas.mpl_connect("key_press_event", self._press_key)
        self.xlim_change_event = self.ax.callbacks.connect('xlim_changed', self._xlim_changed)

        # setup figure
        self._plot()

        

    def print_info(self):
        """
        Print text based user guide in console
        """

        print("\n#===============================================#")
        print("#        How to Use interactive zap mode        #")
        print("#===============================================#\n")
        print("[z]: Enable/Disable zapping mode")
        print("[Left mouse click] (in zapping mode): choose single channel to zap")
        print("[Right mouse click] (in zapping mode): Choose region to zap (zap twice for lower and upper bounds)")
        print("[up arrow key] (in zapping mode): Move the latest zapping line up 1 freq channel unit")
        print("[down arrow key] (in zapping mode): Move the latest zapping line down 1 freq channel unit")
        print("[shift] (in zapping mode): Confirm zapping")
        print("[caps lock] (in zapping mode): Revert most recent zapping")
        print("[control] (in zapping mode): remove all zapping")
        print("[a] (in zapping mode): Automated zapping [default = median method] (requires user input in console)")
        print("[c]: Apply zapping and exit interactive window (this is the only way to apply zapping)\n")
        print("NOTE: Any other way of closing the window beside pressing the [c] key will cancel any zapping!")
        print("#" + "="*100 + "#")


    def _plot(self):

        # plot
        self.zap_ds, self.zap_idx = self._zap_ds()
        self._artists['ds'] = self.ax.imshow(self.zap_ds, aspect = 'auto', 
                                extent = [*self.xlim, self.freqs[-1] - self.df/2, self.freqs[0] + self.df/2],
                                animated = True, interpolation = "none")

        patch_data = np.ones(self.zap_ds.shape[0], dtype = float) * np.nan
        patch_data[self.zap_idx] = 0.55

        self._artists['zapatch'] = self.ax.imshow(patch_data.reshape(patch_data.size, 1),
                                        aspect = 'auto', cmap = 'OrRd', vmax = 1, vmin = 0,
                                        extent = [0.0, 0.02, self.freqs[-1]-self.df/2, self.freqs[0]+self.df/2],
                                        animated = True)
        self.ax.set_xlim(self.xlim)



    def _zap_ds(self):

        self.zapchan = self.base_zapchan

        # get full zapstr
        for i, zaps in enumerate(self.zaps):
            self.zapchan = combine_zapchan(self.zapchan, zaps)
        
        zap_idx = zap_chan(self.freqs, self.zapchan)

        ds = self.ds.copy()
        ds[zap_idx] *= 0.0

        return ds, zap_idx





    def _LMzap(self, event):
        """
        Handle Left mouse click event

        Parameters
        ----------
        event : event.mouse_click
        """

        xlim = self.ax.get_xlim()
        if self._artists['LMline'] is None:
            self._artists['LMline'], = self.ax.plot(xlim, [event.ydata]*2,
                        linestyle = "--", color = "k", linewidth = 1.0, label = "LMline")
        else:
            self._artists['LMline'].set_ydata([event.ydata]*2)

        # set last artist
        self._last_artist = self._artists['LMline']

        # update figure
        self.ax.set_xlim(xlim)
        self.fig.canvas.draw()

        return



    def _RMzap(self, event):
        """
        Handle Right mouse click event

        Parameters
        ----------
        event : event.mouse_click
        """
        xlim = self.ax.get_xlim()
        
        # get line data?
        nlines = len(self._artists['RMlines'])
        if nlines < 2:
            line, = self.ax.plot(xlim, [event.ydata]*2,
                    linestyle = '--', color = "r", linewidth = 1.0, label = f'RMline_{int(nlines+1)}')
            self._artists['RMlines'].append(line)

            self._last_artist = line
        
            
        
        else:
            liney = self._artists['RMlines'][0].get_ydata()[0]
            redraw_idx = 0
            abs_ydist = np.abs(liney - event.ydata)
            if abs(self._artists['RMlines'][1].get_ydata()[0] - event.ydata) < abs_ydist:
                redraw_idx = 1

            # re-draw line with new height
            self._artists['RMlines'][redraw_idx].set_ydata([event.ydata]*2)

            self._last_artist = self._artists['RMlines'][redraw_idx]

        
        # update figure
        self.ax.set_xlim(xlim)
        self.fig.canvas.draw()



    def _xlim_changed(self, event):
        """
        Code to run if x limits have been changed
        """

        if self._artists['zapatch'] is not None:
            new_xlim = event.get_xlim()
            xlim_dif = new_xlim[1] - new_xlim[0]
            self._artists['zapatch'].set(extent = [new_xlim[0], new_xlim[0] + xlim_dif * 0.02, 
                                                   *self._artists['zapatch'].get_extent()[2:]])
            self.ax.draw_artist(self._artists['zapatch'])
        
        return



    def _mouse_press(self, event):
        """
        Handle mouse press event

        Parameters
        ----------
        event : event.mouse_click
        """

        if self.ax.in_axes(event):
            if event.button == MouseButton.LEFT:
                self._LMzap(event)



            elif event.button == MouseButton.RIGHT:
                self._RMzap(event)
        



    

    def _press_key(self, event):
        """
        Process key presses

        Parameters
        ----------
        event : event.key_press
        """
        
        if time.time() - self.last_key_press < RAPIDKEYTHRESHOLD:
            self.last_key_press = time.time()
            return
        # print("key pressed")        
        self.last_key_press = time.time()


        if event.key == "z":
            # Enable or disable zapping
            self.zapmode = ~self.zapmode

            if self.zapmode:
                print("--Enabled zapping mode--")
                # connect mouse events to figure
                self.mouse_click_event = self.fig.canvas.mpl_connect("button_press_event", self._mouse_press)

            else:
                print("--Disabled zapping mode--")
                self.fig.canvas.mpl_disconnect(self.mouse_click_event)

                # remove zapping lines
                if self._artists['LMline'] is not None:
                    self._artists['LMline'].remove()
                    self._artists['LMline'] = None
                
                for i in range(len(self._artists['RMlines'])):
                    self._artists['RMlines'][0].remove()
                    del self._artists['RMlines'][0]
                
                self._last_artist = None
                self.fig.canvas.draw()
            
            self.keys_pressed.add(event.key)

            return
            
        if event.key == "shift":
            if self.zapmode:
                # do zapping
                print("Zapped channel/channels")

                # LM click
                if self._artists['LMline'] is not None:
                    self.zaps.append(str(self._artists['LMline'].get_ydata()[0]))
                    self._artists['LMline'].remove()
                    del self._artists['LMline']
                    self._artists['LMline'] = None
                
                # RM click
                if len(self._artists['RMlines']) == 2:
                    zapbounds = [self._artists['RMlines'][0].get_ydata()[0], 
                                 self._artists['RMlines'][1].get_ydata()[0]]
                    self.zaps.append(f"{np.min(zapbounds)}:{np.max(zapbounds)}")
                for i in range(len(self._artists['RMlines'])):
                    self._artists['RMlines'][0].remove()
                    del self._artists['RMlines'][0]

                self._last_artist = None

                # update figure
                self._update_ax()
            
            return

        if event.key == "caps_lock":
            if self.zapmode:
                # remove prevous zapping
                print("Reverted Previous zapping")
                if len(self.zaps) > 0:
                    self.zaps.pop()

                # update figure
                self._update_ax()

            return

        if event.key == "control":
            if self.zapmode:
                # remove all zapping
                print("Removing all prior zapping")
                self.zaps = []

                # update figure
                self._update_ax()

            return
    
        if event.key == "c":
            # apply zapping
            print("\n#===============================================#")
            print("#       Exiting interactive zapping mode        #")
            print("#===============================================#\n")
            self.final_zapchan = self.zapchan
            plt.close(self.fig)

            return
    
        if event.key == "a":
            # do auto flagging
            if not self.zapmode:
                return
            
            print("automated flagging using method: " + self._auto_alg)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                zap_threshold = input("-[Input]: Value for automated Frequency zapping-\n")
            try:
                zap_threshold = float(zap_threshold)
                print(f"Zapping with a value of {zap_threshold}")
            except:
                print(f"{zap_threshold} is not a valid floating point number!!!")
                return
            
            zaps = self._auto_flag(zap_threshold)

            if zaps is None:
                return

            self.zaps.append(zaps)

            # update time series and freq axes

            # update figure
            self._update_ax()
            self.ax.set_xlim(self.ax.get_xlim())
            
            return
        

        if event.key == "x":
            if not self.zapmode:
                return
            
            # change automatic flagging method
            idx = self._list_of_auto_alg.index(self._auto_alg)
            if idx == len(self._list_of_auto_alg) - 1:
                idx = 0
            else:
                idx += 1
            
            self._auto_alg = self._list_of_auto_alg[idx]

            print("Changed automated flagging method to: " + self._auto_alg)

            return



        if (event.key in ['up', 'down']) and (self._last_artist is not None) and (self.zapmode):
            if event.key == 'up':
                dif = self.df
            else:
                dif = -self.df
            
            # add dif to line y position
            self._last_artist.set_ydata([self._last_artist.get_ydata()[0] + dif]*2)

            # update figure
            self.ax.set_xlim(self.ax.get_xlim())      # this is only here to make sure the figure updates properly
            self.fig.canvas.draw()

            return
        
        return
    


    def _auto_flag(self, zap_threshold):
        """
        Automated algorithm for zapping frequency channels

        """

        xlim = list(self.ax.get_xlim())
        ylim = list(self.ax.get_ylim())

        # transform ylim
        ybounds = [self.freqs[-1] - self.df/2, self.freqs[0] + self.df/2]
        bw = ybounds[1] - ybounds[0]
        ylim[0] = 1.0 - (ylim[0] - ybounds[0])/bw
        ylim[1] = 1.0 - (ylim[1] - ybounds[0])/bw

        tw = self.xlim[1] - self.xlim[0]
        xlim[0] = (xlim[0] - self.xlim[0])/tw
        xlim[1] = (xlim[1] - self.xlim[0])/tw

        ylim = ylim[::-1]

        ds_crop = self.zap_ds.copy()
        ds_crop[self.zap_idx] = np.nan

        ds_crop = pslice(ds_crop, *xlim, axis = 1)
        ds_crop = pslice(ds_crop, *ylim, axis = 0)
        freq_crop = pslice(self.freqs, *ylim)

        # do automated flagging
        mask = np.mean(ds_crop, axis = 1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self._auto_alg == "median":
                median = np.nanmedian(mask)
                mask /= median
                print(f"Median: {median:.2f}")
                zaps = np.where((mask > zap_threshold) & (~np.isnan(mask)))[0]


            elif self._auto_alg == "abs":
                zaps = np.where(mask > zap_threshold)[0]

        if len(zaps) < 1:
            return None

        mask = np.ones(freq_crop.size)
        mask[zaps] = np.nan

        return get_zapstr(mask, freq_crop)


    def _update_ax(self):
        """
        Update dynamic spectrum, frequency and time series and zapping patches
        to display zapped regions
        
        """
        # if xlim and ylim change callbacks are defined!
        # if hasattr(self.ax, 'callbacks'):
        #     _callbacks = self.ax.callbacks.callbacks
        #     if "xlim_changed" in _callbacks.keys():
        #         self.ax.set_xlim(self.ax.get_xlim())
        #     if "ylim_changed" in _callbacks.keys():
        #         self.ax.set_ylim(self.ax.get_ylim())


        self.zap_ds, self.zap_idx = self._zap_ds()
        mask = np.ones(self.zap_ds.shape[0], dtype = bool)
        mask[self.zap_idx] = False
        self._artists['ds'].set(data = self.zap_ds, clim = (np.min(self.zap_ds[mask]), np.max(self.zap_ds[mask])))
        self.ax.draw_artist(self._artists['ds'])

        patch_data = np.ones(self.zap_ds.shape[0]) * np.nan
        patch_data[self.zap_idx] = 0.55
        self._artists['zapatch'].set(data = patch_data.reshape(patch_data.size, 1))
        self.ax.draw_artist(self._artists['zapatch'])

        self.fig.canvas.draw()

        return
    



class _Zoom:
    """
    Class to implement event handles when zooming in on dynamic spectra
    """
    flag_X = False
    flag_Y = False
    new_X = None
    new_Y = None

    def __init__(self, ax):
        ax.callbacks.connect('xlim_changed', self.on_zoom_X)
        ax.callbacks.connect('ylim_changed', self.on_zoom_Y)

        self.ax = ax
        self.new_X = self.ax.get_xlim()
        self.new_Y = self.ax.get_ylim()

    ## update xlims
    def on_zoom_X(zooms,event):
        zooms.flag_X = True
        zooms._update_zoom(event)

    ## update ylims
    def on_zoom_Y(zooms,event):
        zooms.flag_Y = True
        zooms._update_zoom(event)

    ## update profile plots
    def _update_zoom(self_zoom,event):
        if not self_zoom.flag_X or not self_zoom.flag_Y: # only update when both x and y lims have changed
            return
        
        self_zoom.flag_X, self_zoom.flag_Y = False, False


        self_zoom.new_X = event.get_xlim()        # get x lims
        self_zoom.new_Y = event.get_ylim()        # get y lims

        self_zoom.update_zoom()


    def update_zoom(self):
        pass





def ZapInteractive(ds, freqs, times = None, zapchan = None):
    """
    Interactive dynamic spectrum for channel flagging
    
    Parameters
    ----------
    ds : np.ndarray or array-like
        dynamic spectrum
    freqs : np.ndarray or array-like
        frequency array
    times : np.ndarray or array-like
        time array, optional
    zapchan : str
        initial zapchan string (if there is prior zapping applicable)
    
    Returns
    -------
    zapchan : str
        channels to be zap in string format
    """

    class ZapZoom(_Zoom):
        def update_zoom(self):
            if (self.new_X is None) or (self.new_Y is None):
                return
            
            Xphase = list(self.new_X)
            Yphase = [0.0, 1.0]
            bw = freqs[0] - freqs[-1]            
            tw = mouse_zap.xlim[1] - mouse_zap.xlim[0]
            for i in range(2):
                Yphase[i] = 1.0 - (self.new_Y[i] - freqs[-1])/bw
                Xphase[i] = (Xphase[i] - mouse_zap.xlim[0])/tw
            Yphase = Yphase[::-1]

            
            # with new phases, 
            zap_ds_crop = pslice(mouse_zap.zap_ds, *Xphase, axis = 1)
            zap_ds_crop[mouse_zap.zap_idx] = np.nan
            ts_data = np.nanmean(zap_ds_crop, axis = 0)

            zap_ds_crop = pslice(zap_ds_crop, *Yphase, axis = 0)
            fs_data = np.mean(zap_ds_crop, axis = 1)

            ts.set_xdata(pslice(np.linspace(*mouse_zap.xlim, mouse_zap.zap_ds.shape[1]), *Xphase))
            ts.set_ydata(ts_data)
            # ax_ts.set_xlim(self.new_X)
            ts_dif = np.nanmax(ts_data) - np.nanmin(ts_data)
            ax_ts.set_ylim([np.nanmin(ts_data) - 0.05*ts_dif, np.nanmax(ts_data) + 0.05*ts_dif])

            fs.set_data(fs_data, pslice(freqs, *Yphase))
            # ax_fs.set_ylim(self.new_Y)
            fs_dif = np.nanmax(fs_data) - np.nanmin(fs_data)
            ax_fs.set_xlim([np.nanmin(fs_data) - 0.05*fs_dif, np.nanmax(fs_data) + 0.05*fs_dif])

            fig.canvas.draw()
    # addtional event functions for handeling time/freq series data
    # global variables
    global last_key_press
    last_key_press = time.time()

    def _key_press(event):
        global last_key_press

        if time.time() - last_key_press < RAPIDKEYTHRESHOLD:
            last_key_press = time.time()
            return
        # print("key pressed")        
        last_key_press = time.time()

        if event.key in ["shift", "caps_lock", "control"]:
            # replot time and freq series
            zapzoom.update_zoom()

    def _draw_freq_lines(event):


        freq_lines = {}
        freq_line_labels = []
        for line in ax_fs.get_lines():
            line_label = line.get_label()
            if line_label != "freq":
                freq_line_labels += [line_label]
                freq_lines[line_label] = line
        

        # get any lines from ax_ds
        for line in ax_ds.get_lines():
            line_label = line.get_label()
            if line_label not in freq_lines.keys():
                freq_lines[line_label], = ax_fs.plot(ax_fs.get_xlim(), line.get_ydata(),
                                            color = line.get_color(), linestyle = line.get_linestyle(),
                                            linewidth = line.get_linewidth(), label = line_label)
                freq_line_labels += [line_label]
                # freq_lines[line_label].set_xdata(list(ax_fs.get_xlim()))
            
            else:
                if freq_lines[line_label].get_ydata()[0] != line.get_ydata()[0]:
                    # update y line data
                    freq_lines[line_label].set_ydata(line.get_ydata())
        
        ax_ds_lines = []
        for line in ax_ds.get_lines():
            ax_ds_lines += [line.get_label()]
        
        if len(freq_line_labels) < 1:
            return 

        for i in range(len(freq_lines)):
            if freq_line_labels[i] not in ax_ds_lines:
                freq_lines[freq_line_labels[i]].remove()
                del freq_lines[freq_line_labels[i]]


    

    fig, ax = plt.subplots(2, 2, figsize = (10,10), 
                gridspec_kw = {'height_ratios':[1,5],"width_ratios":[6,1]})
    
    ax = ax.flatten()
    
    ax_ts = ax[0]
    ax_ds = ax[2]
    ax_fs = ax[3]
    ax[1].remove()

    ax_ts.get_xaxis().set_visible(False)
    ax_ts.get_yaxis().set_visible(False)
    ax_fs.get_xaxis().set_visible(False)
    ax_fs.get_yaxis().set_visible(False)

    ax_ts.sharex(ax_ds)
    ax_fs.sharey(ax_ds)

    if times is None:
        dynspec_xlim = None
    else:
        dt = times[1] - times[0]
        dynspec_xlim = [times[0] - dt/2, times[-1] + dt/2]
    print(dynspec_xlim)

    mouse_zap = _MouseZapper(fig, ax_ds, ds, freqs, zapchan = zapchan, xlim = dynspec_xlim)
    fig.canvas.mpl_connect("key_press_event", _key_press)
    fig.canvas.mpl_connect("draw_event", _draw_freq_lines)

    zapzoom = ZapZoom(ax_ds)
    
    # plot time and freq series
    ts, = ax_ts.plot([],[], color = 'k', linewidth = 1.5)
    fs, = ax_fs.plot([],[], color = 'k', linewidth = 1.5, label = "freq")

    ax_ds.set_ylabel("Freq [MHz]", fontsize = 16)
    if times is not None:
        ax_ds.set_xlabel("Time [ms]", fontsize = 16)

    zapzoom.update_zoom()

    fig.tight_layout()
    fig.subplots_adjust(hspace = 0, wspace = 0)

    plt.show()

    # code to run after interactive zapping session

    return mouse_zap.final_zapchan
    
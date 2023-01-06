
# %% ***************************************************************************
# IMPORTS
import os

import warnings

import numpy as np

import tkinter as tk

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from hexrd import imageseries


# ***************************************************************************
# CLASS DECLARATION
class pca_parameters_selector_widget_old():
    def __init__(self, pca_paths, pca_mats, adjust_grid=True):
        
        # set up objects
        self.pca_paths = pca_paths
        self.pca_mats = pca_mats
        self.num_det = len(pca_paths.first_img_dict)
        
        # set up image series dict
        self.first_ims_dict = {}
        for det_key in pca_paths.first_img_dict.keys():
            self.first_ims_dict[det_key] = self.pca_mats.load_ims_from_path(self.pca_paths.first_img_dict[det_key])
        
        # set up references
        win_w = 600*self.num_det + 200
        win_h = 800
        fig_w = 600*self.num_det
        fig_h = win_h
        button_w = 160
        button_h = 40
        button_x = win_w - button_w - 20
        
        # set up window and fig
        self.window = tk.Tk()
        self.window.geometry("%ix%i" %(win_w, win_h))
        self.window.title('PCA Parameter Selector')
        self.fig = Figure()
        self.fig_title = "Use the left mouse button to select two corners of a rectangular region of interest"
        self.first_img_ax = self.fig.subplots(nrows=1, ncols=self.num_det)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.get_tk_widget().place(x=0, y=0, height=fig_h, width=fig_w)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.window)
        
        self.bound_box_dict = {}
        self.num_clicks_dict = {}
        for i, det_key in enumerate(self.first_ims_dict.keys()):
            self.bound_box_dict[det_key] = np.empty([2, 2])
            self.num_clicks_dict[det_key] = 0
        
        pd = self.pca_paths.config.material.plane_data
        self.panel_mask_dict = {}
        self.use_rings = tk.IntVar()
        for det_key in pca_mats.det_keys:
            panel = self.pca_paths.config.instrument.hedm.detectors[det_key]
            panel_tth, panel_eta = panel.pixel_angles()
            
            panel_mask = np.zeros(panel_tth.shape)
            for tth in pd.getTThRanges():
                panel_mask += ((panel_tth > tth[0]) & (panel_tth < tth[1]))
            
            self.panel_mask_dict[det_key] = panel_mask
        
        self.update_plot()
        
        # handle mouse cursor button for figure
        def on_fig_cursor_press(event):
            if event.inaxes is not None and str(event.button) == 'MouseButton.LEFT' and str(self.toolbar.mode) != 'zoom rect':
                for i, det_key in enumerate(self.first_ims_dict.keys()):
                    if event.inaxes == self.first_img_ax[i]:
                
                        if self.num_clicks_dict[det_key] == 0:
                            # update plot here with on point scatter
                            self.bound_box_dict[det_key][0, :] = [event.xdata, event.ydata]
                            self.first_img_ax[i].scatter(event.xdata, event.ydata, c='r', s=50, marker='.')
                            self.canvas.draw()
                        else:
                            self.bound_box_dict[det_key][1, :] = [event.xdata, event.ydata]
                            self.bound_box_dict[det_key] = np.sort(self.bound_box_dict[det_key], axis=0)
                            
                            if self.pca_mats.box_points_dict[det_key].size == 0:
                                self.pca_mats.box_points_dict[det_key] = np.copy(self.bound_box_dict[det_key].reshape([1, 2, 2]))
                            else:
                                self.pca_mats.box_points_dict[det_key] = np.vstack([self.pca_mats.box_points_dict[det_key], 
                                                                                    np.copy(self.bound_box_dict[det_key].reshape([1, 2, 2]))])
                            self.update_plot()
                        
                        self.num_clicks_dict[det_key] = (self.num_clicks_dict[det_key] + 1) % 2
    
        self.fig.canvas.mpl_connect('button_press_event', on_fig_cursor_press)
        
        # Add a button for loading pca paramters
        # TODO: Only load ROI file, not entire object...?
        def load_pca_matrices_on_click():
            print("Load RealTimePCA Matrices")
            root = tk.Tk()
            root.withdraw()
            
            pca_mats_dir = tk.filedialog.askopenfilename(initialdir=self.pca_paths.output_dir,
                                                           title='Select RealTimePCA Matrices File')
            
            if not pca_mats_dir:
                pass
            else:
                try:
                    self.pca_mats.load_pca_matrices_from_file(pca_mats_dir)
                    self.update_plot()
                except Exception as e:
                    print(e)
                    warnings.warn("Something went wrong when loading RealTimePCA Matrices file")
                    pass
            
            # update sliders
        self.load_pca_mats_button = tk.Button(self.window, text="Load ROI", command=load_pca_matrices_on_click)
        self.load_pca_mats_button.place(x=button_x, y=200, height=button_h, width=button_w)
        
        # Add a button for saving pca parameters
        # TODO: Only save ROI file, not entire object...?
        def save_pca_matrices_on_click():
            print("Save RealTimePCA Matrices")
            root = tk.Tk()
            root.withdraw()
            
            pca_mats_dir = tk.filedialog.asksaveasfilename(confirmoverwrite=False,
                                                                 initialdir=self.pca_paths.output_dir,
                                                                 title='Save RealTimePCA Matices')
            
            if not pca_mats_dir:
                pass
            else:
                try:
                    self.pca_mats.save_pca_matrices_to_file(pca_mats_dir)
                except Exception as e:
                    print(e)
                    warnings.warn("Something went wrong when saving RealTimePCA Matrices file")
                    pass
            
        self.save_pca_mats_button = tk.Button(self.window, text="Save ROI", command=save_pca_matrices_on_click)
        self.save_pca_mats_button.place(x=button_x, y=250, height=button_h, width=button_w)
        
        # Add a button for clearing boxes
        def clear_boxes():
            self.pca_mats.reset_box_points_dict()
            self.update_plot()
            
        self.clear_button = tk.Button(self.window, text="Clear Boxes", command=clear_boxes)
        self.clear_button.place(x=button_x, y=300, height=button_h, width=button_w)
        
        # Add a checkbox for using rings
        def use_rings_on_click():
            self.use_rings.set((self.use_rings.get() + 1) % 2)
            
            # !!!TODO: this should be made better, maybe a flag in pca_mats to use or not, and not resetting a potentially large variable
            if self.use_rings.get() == 1:
                self.pca_mats.rings_mask_dict = self.panel_mask_dict
            else:
                self.pca_mats.rings_mask_dict = {}
            self.update_plot()
            
        self.use_rings_check = tk.Checkbutton(self.window, text='Use Rings', variable=self.use_rings, 
                                              onvalue=1, offvalue=0, command=use_rings_on_click)
        self.use_rings_check.place(x=button_x, y=350, height=button_h, width=button_w)

        
        # Add a button for quitting
        def on_closing(root):
            root.destroy()
            root.quit()            
        self.quit_button = tk.Button(self.window, text="Continue", command=lambda root=self.window:on_closing(root))
        self.quit_button.place(x=button_x, y=700, height=button_h, width=button_w)
        self.window.protocol("WM_DELETE_WINDOW", lambda root=self.window:on_closing(root))
        self.window.mainloop()
        
    
        
    def update_plot(self):
        self.fig.suptitle(self.fig_title)
        for i, det_key in enumerate(self.first_ims_dict.keys()):
            self.first_img_ax[i].clear()
            self.first_img_ax[i].imshow(self.first_ims_dict[det_key][0], vmax=100)
            
            for j in range(self.pca_mats.box_points_dict[det_key].shape[0]):             
                bbc = self.pca_mats.box_points_dict[det_key][j, :, :]
                selected_rect = patches.Rectangle((bbc[0,0],bbc[0,1]), bbc[1,0]-bbc[0,0], 
                                        bbc[1,1]-bbc[0,1],linewidth=1, edgecolor='r', facecolor='none')
                self.first_img_ax[i].add_patch(selected_rect)
            
            if self.use_rings.get() == 1:
                self.first_img_ax[i].imshow(self.panel_mask_dict[det_key], vmax=1, cmap='Reds', alpha=0.1)
            
        self.canvas.draw()
    
    def get_all_pca_objects(self):
        return [self.pca_paths, self.pca_mats]

class pca_parameters_selector_widget():
    def __init__(self, lodi_exp, vmax=1000, img_process_dict=None):
        
        # set up objects
        self.lodi_exp = lodi_exp
        self.num_det = len(self.lodi_exp.det_keys())
        
        # set up image series dict
        self.first_ims_dict = {}
        for det_key in self.lodi_exp.det_keys():
            self.first_ims_dict[det_key] = self.lodi_exp.load_ims_from_path(self.lodi_exp.first_img_dict[det_key], 
                                                                            img_process_list=img_process_dict[det_key])
        
        # set up references]]
        win_w = 600*self.num_det + 200
        win_h = 800
        fig_w = 600*self.num_det
        fig_h = win_h
        button_w = 160
        button_h = 40
        button_x = win_w - button_w - 20
        
        # set up window and fig
        self.window = tk.Tk()
        self.window.geometry("%ix%i" %(win_w, win_h))
        self.window.title('PCA Region of Interest Selector')
        self.fig = Figure()
        self.fig_title = "Use the left mouse button to select two corners of a rectangular region of interest"
        self.first_img_ax = self.fig.subplots(nrows=1, ncols=self.num_det)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.get_tk_widget().place(x=0, y=0, height=fig_h, width=fig_w)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.window)
        
        # set up box mask variables
        self.bound_box_dict = {}
        self.num_clicks_dict = {}
        for i, det_key in enumerate(self.lodi_exp.det_keys()):
            self.bound_box_dict[det_key] = np.empty([2, 2])
            self.num_clicks_dict[det_key] = 0
        
        # set up ring mask variables
        self.use_rings = tk.IntVar()
        
        # set up plotting variables
        self.vmax = vmax
        
        # start plotting
        self.update_plot()
        
        
        # GUI FUNCTIONS *******************************************************
        
        # handle mouse cursor button for figure
        def on_fig_cursor_press(event):
            if event.inaxes is not None and str(event.button) == 'MouseButton.LEFT' and str(self.toolbar.mode) != 'zoom rect':
                for i, det_key in enumerate(self.lodi_exp.det_keys()):
                    if event.inaxes == self.first_img_ax[i]:
                
                        if self.num_clicks_dict[det_key] == 0:
                            # update plot here with on point scatter
                            self.bound_box_dict[det_key][0, :] = [event.xdata, event.ydata]
                            self.first_img_ax[i].scatter(event.xdata, event.ydata, c='r', s=50, marker='.')
                            self.canvas.draw()
                        else:
                            self.bound_box_dict[det_key][1, :] = [event.xdata, event.ydata]
                            self.bound_box_dict[det_key] = np.sort(self.bound_box_dict[det_key], axis=0)
                            
                            bbd_copy = np.copy(self.bound_box_dict[det_key].reshape([1, 2, 2]))
                            if self.lodi_exp.box_points_dict[det_key].size == 0:
                                self.lodi_exp.box_points_dict[det_key] = bbd_copy
                            else:
                                self.lodi_exp.box_points_dict[det_key] = np.vstack([self.lodi_exp.box_points_dict[det_key], 
                                                                                    bbd_copy])
                            self.update_plot()
                        
                        self.num_clicks_dict[det_key] = (self.num_clicks_dict[det_key] + 1) % 2
    
        self.fig.canvas.mpl_connect('button_press_event', on_fig_cursor_press)
        
        # Add a button for loading pca paramters
        # TODO: Only load ROI file, not entire object...?
        def load_masks_on_click():
            self.lodi_exp.load_mask_dict_from_file()
            self.update_plot()
            
        self.load_pca_mats_button = tk.Button(self.window, text="Load Masks", command=load_masks_on_click)
        self.load_pca_mats_button.place(x=button_x, y=200, height=button_h, width=button_w)
        
        # Add a button for saving pca parameters
        # TODO: Only save ROI file, not entire object...?
        def save_masks_on_click():
            self.lodi_exp.save_mask_dict_to_file()
            
        self.save_pca_mats_button = tk.Button(self.window, text="Save Masks", command=save_masks_on_click)
        self.save_pca_mats_button.place(x=button_x, y=250, height=button_h, width=button_w)
        
        # Add a button for clearing boxes
        def clear_boxes():
            self.lodi_exp.reset_box_points_dict()
            self.update_plot()
            
        self.clear_button = tk.Button(self.window, text="Clear Boxes", command=clear_boxes)
        self.clear_button.place(x=button_x, y=300, height=button_h, width=button_w)
        
        # Add a checkbox for using rings
        def use_rings_on_click():
            self.use_rings.set((self.use_rings.get() + 1) % 2)
            for det_key in self.lodi_exp.det_keys():
                self.lodi_exp.use_rings_mask_dict[det_key] = self.use_rings.get()
            self.update_plot()
            
        self.use_rings_check = tk.Checkbutton(self.window, text='Use Rings', variable=self.use_rings, 
                                              onvalue=1, offvalue=0, command=use_rings_on_click)
        self.use_rings_check.place(x=button_x, y=350, height=button_h, width=button_w)
        
        # Add a button for quitting
        def on_closing(root):
            tot_mask_dict = self.lodi_exp.total_img_mask_dict()
            for det_key in self.lodi_exp.det_keys():
                u_pix = np.sum(tot_mask_dict[det_key])
                t_pix = tot_mask_dict[det_key].size
                perc = float(u_pix) / float(t_pix)
                print("%s: using %i pixels / %i total pixels (%0.2f)" %(det_key, u_pix, t_pix, perc))
            
            root.destroy()
            root.quit()            
        self.quit_button = tk.Button(self.window, text="Continue", command=lambda root=self.window:on_closing(root))
        self.quit_button.place(x=button_x, y=700, height=button_h, width=button_w)
        self.window.protocol("WM_DELETE_WINDOW", lambda root=self.window:on_closing(root))
        self.window.mainloop()
        
        
    def update_plot(self):
        self.fig.suptitle(self.fig_title)
        tot_mask = self.lodi_exp.total_img_mask_dict()
        for i, det_key in enumerate(self.first_ims_dict.keys()):
<<<<<<< HEAD
            if len(self.first_ims_dict.keys()) > 1:
                self.first_img_ax[i].clear()
                self.first_img_ax[i].title.set_text(det_key)
                self.first_img_ax[i].imshow(self.first_ims_dict[det_key][0], vmax=self.vmax)
                self.first_img_ax[i].imshow(tot_mask[det_key], vmax=1, cmap='Reds', alpha=0.1)
            else:
                self.first_img_ax.clear()
                self.first_img_ax.title.set_text(det_key)
                self.first_img_ax.imshow(self.first_ims_dict[det_key][0], vmax=self.vmax)
                self.first_img_ax.imshow(tot_mask[det_key], vmax=1, cmap='Reds', alpha=0.1)
=======
            self.first_img_ax[i].clear()
            self.first_img_ax[i].title.set_text(det_key)
            self.first_img_ax[i].imshow(self.first_ims_dict[det_key][0], vmax=self.vmax)
            self.first_img_ax[i].imshow(tot_mask[det_key], vmax=1, cmap='Reds', alpha=0.1)
>>>>>>> 3b0ecb6565d08a5e078485c2ec88f34e0b61bdf9
            
        self.canvas.draw()
    
    def get_all_pca_objects(self):
        return [self.lodi_exp]
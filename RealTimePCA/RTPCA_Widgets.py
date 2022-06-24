import os
import numpy as np

import tkinter as tk

import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


# ***************************************************************************
# CLASS DECLARATION
class pca_parameters_selector_widget():
    def __init__(self, pca_paths, pca_mats, adjust_grid=True):
        
        self.pca_paths = pca_paths
        self.pca_mats = pca_mats
        
        self.window = tk.Tk()
        self.window.geometry("1000x800")
        self.window.title('pca Parameter Selector')
        self.first_img = cv2.imread(pca_paths.get_img_num_dir(pca_paths.first_img_num), 0)
        
        self.fig = Figure(figsize=(6,6))
        self.fig.suptitle("Use the left mouse button to select a region of interest")
        self.first_img_ax = self.fig.add_subplot(111)
        self.first_img_ax.imshow(self.first_img, cmap='Greys_r')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.get_tk_widget().place(x=0, y=0, height=800, width=800)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.window)
        self.upper_left_points = np.array([])
        self.lower_right_points = np.array([])
        
        self.bound_box_array = self.pca_mats.calc_bounding_box()
        
        # handle mouse cursor button for figure grid selection
        if adjust_grid:
            self.num_clicks = 0
            def on_fig_cursor_press(event):
                if event.inaxes is not None and str(event.button) == 'MouseButton.LEFT' and str(self.toolbar.mode) != 'zoom rect':
                    if self.num_clicks == 0:
                        self.bound_box_array[0, :] = [event.xdata, event.ydata]
                        self.upper_left_points = np.append(self.upper_left_points, (event.xdata,event.ydata))
                        self.first_img_ax.scatter(event.xdata, event.ydata, c='r', s=50, marker='.')
                        self.canvas.draw()
                        # update plot here with on point scatter
                    else:
                        self.bound_box_array[1, :] = [event.xdata, event.ydata]
                        self.lower_right_points = np.append(self.lower_right_points, (event.xdata,event.ydata))

                        self.update_plot()
                        # update plot here with on point scatter and with grid
                    
                    self.num_clicks = (self.num_clicks + 1) % 2
        
            self.fig.canvas.mpl_connect('button_press_event', on_fig_cursor_press)
        
        if not adjust_grid and self.pca_mats.ref_points.size != 0:
            self.update_plot()
        
        # Add a button for quitting
        def on_closing(root):
            root.destroy()
            root.quit()            
        self.quit_button = tk.Button(self.window, text="Quit", command=lambda root=self.window:on_closing(root))
        self.quit_button.place(x=820, y=700, height=40, width=160)
        self.window.protocol("WM_DELETE_WINDOW", lambda root=self.window:on_closing(root))
        self.window.mainloop()
    
        
    def update_plot(self):
        # initialize local variables
        bbc = self.bound_box_array

        selected_rect = patches.Rectangle((bbc[0, 0],bbc[0,1]), bbc[1,0]-bbc[0,0], 
                                bbc[1,1]-bbc[0,1],linewidth=1, edgecolor='r', facecolor='none')
        
        
        # Add the patch to the Axes
        self.pca_mats.box_points = np.vstack((self.upper_left_points,self.lower_right_points))
        self.fig.suptitle("Use the left mouse button to select two corners of a rectangular region of interest")
        self.first_img_ax.imshow(self.first_img, cmap='Greys_r')
        self.first_img_ax.add_patch(selected_rect)
        self.canvas.draw()
    
    def get_all_pca_objects(self):
        return [self.pca_paths, self.pca_mats]
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Nathan & Nicolas
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

class FIP_Plots():
    def __init__(self, freq_radday = np.ones(1), fips = np.ones(1),
                       starname = None):

        assert len(freq_radday) == len(fips), "Frequency array and FIP array must be the same length"

        #Peaks of the smoothed lasso solution
        self.omega_peaks = None  # frequency of the peaks
        self.peakvalues = None # height of the peaks
        self.mlog10fip =  -np.log10(fips) 
        self.omegas = freq_radday 
        self.fips = fips
        self.periods = 2*np.pi/freq_radday 
        self.ltip = np.log10(1-fips)
        if starname == None:
            self.starname =  ''
        else:
            self.starname = starname

        self.fiplevel = None
        self.truepos = None
        self.bernouilli_unc = None
            
    def find_peaks(self):
        '''
        Define the peaks appearing in the smoothed solution
        solution: their frequencies self.omega_peaks
        their values self.peakvalues, and decreasing order
        self.indexsort
        '''
        i = 0

        peakpos = []
        Nomegas = len(self.omegas)
        
        while i<Nomegas-1:
            if self.mlog10fip[i] > 1e-2:
                j = i
                while self.mlog10fip[i] > 1e-2 and i<Nomegas-1:
                    i = i+1
                array = self.mlog10fip[j:i]
                peakpos1 = j+np.argmax(array)
                peakpos.append(peakpos1)
            i = i+1

    
        peakvals = self.mlog10fip[peakpos]
        indexsort = np.argsort(-peakvals)
        peakpos = np.array(peakpos)
        peakpos_sort = peakpos[indexsort]
        
        if len(peakpos_sort)>0:
            self.omega_peaks = self.omegas[peakpos_sort]
            self.peakvalues = self.mlog10fip[peakpos_sort]
        else:
            self.omega_peaks = []
            self.peakvalues = []
            
        return self.omega_peaks, self.peakvalues
            

    def plot_clean(self, number_highlighted_peaks_in, maxpla,
                   fip_orientation = 'up',
                   fip_label = None,
                   alpha=1,
                   plot_tip=True,
                   annotations='periods',
                   marker_color = (0.85,0.325,0.098),
                   colorfip=(0,0.447,0.741),
                   colortip=(0.9290,0.6940,0.1250),
                   title = 'default',
                   save = False,
                   save_tag = None,
                   **kwargs): 
        ''' 
        Plot the FIP periodogram with highlighted highest peaks
        INPUTS:
            - number_highlighted_peaks_in: number of peaks to highlight
            - annotations: labels of the peaks 
            annotations is either set to a key of the 
            dictionary self.significance, and then corresponds to the
            significance of the highlighted peaks or to 'periods',
            in which case it shows the peak periods
            - marker_color: color of the marhers that highlight the 
            highest peaks
            - title: plot title
            - save: saves a pdf file if set to True
        '''
        #start_time = time.time()
        
        self.find_peaks()
        
        nmaxpeaks = len(self.omega_peaks)
        number_highlighted_peaks = min(number_highlighted_peaks_in,nmaxpeaks)
        if number_highlighted_peaks_in >  nmaxpeaks:
            print('There are only', nmaxpeaks, 'peaks')
            
        peakvalues_plot = self.peakvalues[:number_highlighted_peaks]
        
        if title=='default':
            fipperio_title = f'{self.starname} FIP periodogram'
        else:
            fipperio_title = title
        
        if len(self.omega_peaks[:number_highlighted_peaks])>0:
            periods_plot = 2*np.pi/self.omega_peaks[:number_highlighted_peaks]
        else:
            periods_plot =[]
        

        # bm = (0,0.447,0.741)
        # gr = (0.2,0.4,0.)
        # rr = (0.6,0.,0.6)#(0.8,0.447,0.741)
        # gy = (0.9290,0.6940,0.1250)
        # rm = (0.85,0.325,0.098)

        if number_highlighted_peaks>0:
            periods_maxpeaks = periods_plot
        else:
            periods_maxpeaks = 0
        maxperiod = self.periods[0]

        # Create figures
        fig, ax1 = plt.subplots(figsize=(10, 4))
        # colorfip = bm
        # colortip = gy
     
        # Plot TIP
        if plot_tip:
            ax2 = ax1.twinx() # Create another axis to plot the TIP
            ax2.semilogx(self.periods,self.ltip, alpha=0.7,
                        color=colortip, linewidth=1.2, zorder=-1)
            ax2.set_ylabel(r'$\log_{10}$ TIP', color=colortip, fontsize=18)

            # Set y limits
            ylim2 = ax2.get_ylim()
            ax2.set_ylim([ylim2[0],0])        
        
        # Plot FIP
        if fip_orientation == 'up':
            ax1.plot(self.periods, self.mlog10fip, 
                     linewidth=1.7, color=colorfip, alpha=alpha,
                     label=fip_label)
            ax1.set_xscale('log')
            ax1.set_ylim([0, np.max(self.mlog10fip)*1.15])
            ax1.set_ylabel(r'-$\log_{10}$ FIP', fontsize=18, color=colorfip)
            ax1.set_zorder(1) # Place on top of TIP
            ax1.patch.set_visible(False)
        else:
            ax1.plot(self.periods, -self.mlog10fip, 
                     linewidth=1.7, color=colorfip, alpha=alpha,
                     label=fip_label)
            ax1.set_ylim([np.min(-self.mlog10fip)*1.15,0])
            ax1.set_ylabel(r'$\log_{10}$ FIP', fontsize=18, color=colorfip)
            peakvalues_plot = - peakvalues_plot
                
        # Set x axis limits               
        minP = min(self.periods)
        maxP = max(self.periods)
        ax1.set_xlim([minP,maxP])
        
        # Get width and height of final plot
        ylim = ax1.get_ylim()
        deltaY = ylim[1] - ylim[0]
        deltaX = np.log10(maxP) - np.log10(minP)

        # Plot markers with highest peaks
        if number_highlighted_peaks>0:
            if annotations is None:
                point_label=''          
            elif annotations=='periods':
                point_label = 'Peak periods (d)'
            else:
                raise Exception(('The annotations key word '
                          'has to be a key of the self.significance dictionary '
                          'or ''periods'' or None'))
                
            # Cleaner outputs for standard significance evaluations
            if annotations=='log10faps':
                point_label = r'$\log_{10}$' + ' FAPs'
            if annotations=='log_bayesf_laplace':
                point_label = r'$\log$' + ' Bayes factor'
            if annotations=='log10_bayesf_laplace':
                point_label = r'$\log_{10}$' + ' Bayes factor'
            
            # Plot markers on highest peaks
            ax1.plot(periods_maxpeaks, peakvalues_plot,'o',
                     markersize=5, color=marker_color, label = point_label)
        
        
        y_legend = np.zeros(number_highlighted_peaks)
        x_legend = np.zeros(number_highlighted_peaks)

        for j in range(number_highlighted_peaks):
            
            annotation = ''
            if annotations == 'periods':
                per = periods_maxpeaks[j]   
                p = str(per)
                indexpoint = p.find('.')
                if indexpoint<3:
                    ndigits = 4
                else:
                    ndigits = indexpoint
                per = round(per, ndigits - indexpoint)
                p = str(per)
                annotation = p[:ndigits]
                
            elif annotations is not None:
                if j < len(self.significance[annotations]):
                    annotation = sci_notation(self.significance[annotations][j], decimal_digits=1, 
                                     precision=None, exponent=None)
                else:
                    annotation=''
            
            x1 = periods_maxpeaks[j]*1.17
            y1 = peakvalues_plot[j] #- deltaY/10
            if np.log10(np.abs(x1/maxperiod))>0.85:
                x1 = periods_maxpeaks[j]*0.8
            #if np.log10((x1 - minperiod)/minperiod)<0.1
            
            #deltalogp = np.lod10(max(periods) - min(periods))
            diff_y = np.abs(y1 - y_legend[:j])
            diff_x = np.abs((np.log10(x1/x_legend[:j])))
            
            condition = (diff_y <deltaY/9) *  (diff_x<deltaX/9)
            index_cond = [i for i,v in enumerate(condition) if v]
            
   
            indices = np.arange(j)
            if j>0 and np.sum(condition)>0:
                distx = diff_x[index_cond]
                ii = indices[condition][np.argmin(distx)]
                if y1 <y_legend[ii]:
                    if y1>deltaY/10:
                        y1 = y_legend[ii] - deltaY/10
                    else:
                        x1 = x1* 1.1
                else:
                    y1 = y_legend[ii] + deltaY/10
            y_legend[j] = y1 
            x_legend[j] = x1                 
            
            if annotation is not None:
                ax1.annotate(annotation, (x1,y1),
                             ha='left', fontsize=14, 
                             color=(0.85,0.325,0.098),
                             bbox=dict(facecolor='white', 
                                       edgecolor = 'white',
                                       alpha=0.8))         
        
        ax1.tick_params(axis='y', colors=colorfip)
        #ax2.tick_params(axis='y', colors=colortip)
        ax1.set_xlim([self.periods[0], self.periods[-1]])
        #ax2.set_ylim([-8, 0 ])     
        ax1.set_xlabel('Period (days)', fontsize=18)     
        #plt.suptitle('FIP periodogram',fontsize=20,y=0.97)
        # ax1.legend(fontsize = 16, loc='best')
        plt.title(fipperio_title ,fontsize=20,y=1.05)   
        plt.tight_layout()
        
        print(f"In plot I have Maxpla={maxpla}")
        if save:
            string_save = self.starname.replace(' ', '_') + f'_FIP_periodogram_maxpla{maxpla}'
            if save_tag is not None:
                assert type(save_tag) == str, "save_tag has to be a string"
                string_save += save_tag
            plt.savefig(string_save + '.pdf', format='pdf')
            plt.savefig(string_save + '.png', format='png', dpi=150)

        return fig, ax1, periods_maxpeaks, peakvalues_plot
            
    def plot_multiple(self, fip_arrays, highlighted_peaks, labels=None, alpha=0.6):
        # Define colormap
        color_map = iter(plt.get_cmap('Dark2').colors)

        fip_arrays = np.atleast_2d(fip_arrays) # To handle single arrays
        if labels is not None:
            fig, ax1, _, _ = self.plot_clean(highlighted_peaks, annotations=None,
                                             fip_label=labels[0], alpha=alpha,
                                             plot_tip=False, colorfip=next(color_map))
            for i in range(1, len(fip_arrays)):
                ax1.plot(self.periods, -np.log10(fip_arrays[i]), label=labels[i],
                         linewidth=1.7, color=next(color_map), alpha=alpha)

            ax1.set_ylim([0, np.max(-np.log10(fip_arrays))*1.15])
            ax1.set_ylabel(r'-$\log_{10}$ FIP', fontsize=18, color='k')
            ax1.tick_params(axis='y', colors='k')
            ax1.legend()

        else:
            fig, ax1, _, _ = self.plot_clean(highlighted_peaks, annotations=None, alpha=alpha)
            for i in range(1, len(fip_arrays)):
                ax1.plot(self.periods, -np.log10(fip_arrays[i]), 
                         linewidth=1.7, color=next(color_map), alpha=alpha)

        filename = f"{self.starname.replace(' ', '')} _FIP_several_runs.png"
        plt.savefig(filename, dpi=150)

            
    def plot_alpha(self, save=True, 
                   suffix = '',zoom_factor=2/5, path=''):
            
        x = self.fiplevel
        ys = self.truepos
        errs = self.bernouilli_unc
        nsim = ys.shape[0]

        diffs = np.zeros_like(ys)
        for i in range(nsim):
            diffs[i,:] = ys[i,:] - x
        
        maxdif = np.max(diffs)
        mindif = np.min(diffs)
        ratio = (maxdif-mindif)*zoom_factor
        

        fig, (ax1, ax2) = plt.subplots(2, 1, 
             figsize=(6,6),sharex=True,gridspec_kw={
                           'height_ratios': [1,zoom_factor]})        
        
        #ax1.set_aspect('equal')
        #ax2.set_aspect('equal')
        ax1.set_xlim(-0.05,1.05)
        #ax1.set_ylim(-0.05,1.05)
        

        #plt.subplots(1, 2, figsize=(10, 4.2))
      
        ax2.set_xlabel('True inclusion probability (TIP)' 
                   ,fontsize=16)
        
        ax1.set_ylabel('Fraction of success', fontsize=16)
        
        ax2.set_ylabel('Fration of success\nnormalized', fontsize=16) 
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax2.tick_params(axis='both', which='major', labelsize=14)
        fig.suptitle('Fraction of success vs. TIP',fontsize=18)
        
        ax1.plot([0,1], [0,1], linestyle ='--',color='black', 
                 label = 'y = x')
        ax2.plot([0,1], [0,0], linestyle ='--',color='black')
        
        for i in range(nsim):
            ax1.errorbar(x, ys[i,:], errs[i,:], fmt='o', alpha=0.7)
            # diff = ys[i,:] - x
            # ax2.errorbar(x, diffs[i]/errs[i], errs[i,:], fmt='o', alpha=0.7)
            ax2.plot(x, diffs[i]/errs[i], 'o', alpha=0.7)
        ax1.legend(fontsize=14)
            
        plt.gcf().subplots_adjust(left=0.23)
        if save:
            plt.savefig(os.path.join(path,f'alphacheck_{suffix}.pdf'), fmt='pdf')


def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    written by sodd (stackoverflow user)
    
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    if num!=0 and ~np.isnan(num):
        if exponent is None:
            exponent = int(np.floor(np.log10(abs(num))))
        coeff = round(num / float(10**exponent), decimal_digits)
        if precision is None:
            precision = decimal_digits
        if exponent !=0:
            return r"${0:.{2}f}\cdot10^{{{1:d}}}$".format(coeff, exponent, precision)
        else:
            return r"${0:.{2}f}$".format(coeff, exponent, precision)
    else:
        return('0')   

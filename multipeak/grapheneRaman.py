from . import *

import numpy as np

from .multipeak import makePlotLabel, FitResultParameter, MultiPeakModelResults

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import savefig

class GrapheneModelResults(MultiPeakModelResults):
    def __init__(self, fittedModel, dataset):
        super(GrapheneModelResults, self).__init__(fittedModel, dataset)
        self.widthD = self.peakResults.getReferenceByName('D_fwhm')
        self.widthG = self.peakResults.getReferenceByName('G_fwhm')
        self.widthtwoD = self.peakResults.getReferenceByName('twoD_fwhm')
        
        self.centerD = self.peakResults.getReferenceByName('D_center')
        self.centerG = self.peakResults.getReferenceByName('G_center')
        self.centertwoD = self.peakResults.getReferenceByName('twoD_center')
        
        self.areaG = self.peakResults.getReferenceByName('G_amplitude')

        self.DtoG_area = self.calcRatio('D_amplitude', 'G_amplitude', min=0.01, max=5.0, outliers=5)
        self.DtoG_area.description = 'D/G area ratio'
        self.DtoG_height = self.calcRatio('D_height', 'G_height', min=0.01, max=5.0, outliers=5)
        self.DtoG_height.description = 'D/G height ratio'
        self.twoDtoG_area = self.calcRatio('twoD_amplitude', 'G_amplitude', min=0.01, max=10.0, outliers=5)
        self.twoDtoG_area.description = '2D/G area ratio'
        self.twoDtoG_height = self.calcRatio('twoD_height', 'G_height', min=0.01, max=10.0, outliers=5)
        self.twoDtoG_height.description = '2D/G height ratio'
        
        self.twoDmaxRes = FitResultParameter(np.ones_like(self.x[:,0]))
        self.twoDmaxRes.description = 'Normalized squared maximum 2D residual'
        self.twoDr2 = FitResultParameter(np.ones_like(self.x[:,0]))
        self.twoDr2.description = 'Coefficient of determination of 2D single peak fitting'
        
        self.derivedResults['DtoG_area'] = self.DtoG_area
        self.derivedResults['DtoG_height'] = self.DtoG_height
        self.derivedResults['twoDtoG_area'] = self.twoDtoG_area
        self.derivedResults['twoDtoG_height'] = self.twoDtoG_height
        self.derivedResults['twoDmaxRes'] = self.twoDmaxRes
        self.derivedResults['twoDr2'] = self.twoDr2
        
        self.get2Dresiduals()
            
    def writeOutput(self, filenameAd):
        if self.datasetsNumber < 1:
            return
        
        out = np.ndarray((len(self.x[:,0]), 20))
        header = ''
        delimiter = ';'
        
        fittedD =  self.fitResults[0].best_values.get('D_center') != None
        fittedG =  self.fitResults[0].best_values.get('G_center') != None
        
        fittedtwoD =  self.fitResults[0].best_values.get('twoD_center') != None
        
        for i, xdata in enumerate(self.x[:,0]):
            j = 0
            if i == 0: header += 'Position x'
            out[i][j] = self.posX[i,0]
            j += 1
            if i == 0: header += delimiter + 'Position y'
            out[i][j] = self.posY[i,0]
            
            if fittedD:
                j += 1
                if i == 0: header += delimiter + 'D center'
                out[i][j] = self.fitResults[i].params['D_center']
                j += 1
                if i == 0: header += delimiter + 'D area'
                out[i][j] = self.fitResults[i].params['D_amplitude']
                j += 1
                if i == 0: header += delimiter + 'D FWHM'
                out[i][j] = self.fitResults[i].params['D_fwhm']
                j += 1
                if i == 0: header += delimiter + 'D fraction'
                out[i][j] = self.fitResults[i].params['D_fraction']
                
            if fittedG:
            
                j += 1
                if i == 0: header += delimiter + 'G center'
                out[i][j] = self.fitResults[i].params['G_center']
                j += 1
                if i == 0: header += delimiter + 'G area'
                out[i][j] = self.fitResults[i].params['G_amplitude']
                j += 1
                if i == 0: header += delimiter + 'G FWHM'
                out[i][j] = self.fitResults[i].params['G_fwhm']
                j += 1
                if i == 0: header += delimiter + 'G fraction'
                out[i][j] = self.fitResults[i].params['G_fraction']
            
            if fittedtwoD:
                j += 1
                if i == 0: header += delimiter + '2D center'
                out[i][j] = self.fitResults[i].params['twoD_center']
                j += 1
                if i == 0: header += delimiter + '2D area'
                out[i][j] = self.fitResults[i].params['twoD_amplitude']
                j += 1
                if i == 0: header += delimiter + '2D FWHM'
                out[i][j] = self.fitResults[i].params['twoD_fwhm']
                j += 1
                if i == 0: header += delimiter + '2D fraction'
                out[i][j] = self.fitResults[i].params['twoD_fraction']
            
            if fittedD:
                if fittedG:
                    j += 1
                    if i == 0: header += delimiter + 'D/G (area)'
                    out[i][j] = self.DtoG_area.values[i]
                    j += 1
                    if i == 0: header += delimiter + 'D/G (height)'
                    out[i][j] = self.DtoG_height.values[i]
                
            if fittedtwoD:
                if fittedG:
                    j += 1
                    if i == 0: header += delimiter + '2D/G (area)'
                    out[i][j] = self.twoDtoG_area.values[i]
                    
                    j += 1
                    if i == 0: header += delimiter + '2D/G (height)'
                    out[i][j] = self.twoDtoG_height.values[i]
           
                j += 1
                if i == 0: header += delimiter + '2D NSR'
                out[i][j] = self.twoDmaxRes.values[i]
                j += 1
                if i == 0: header += delimiter + '2D R2'
                out[i][j] = self.twoDr2.values[i]
            
        j += 1
        out.resize(i,j)
        
        header += delimiter + str(self.task)
        filename = self.baseFilename + '_' + filenameAd + '.csv'
        np.savetxt(filename, out, delimiter=delimiter, header=header)
        print('Results for each sprectrum saved as: ', filename)    
        
    def printAvgRes(self):
        if self.datasetsNumber < 1:
            return
        out = str(self.DtoG_area.avg)
        out += ' ' + str(self.DtoG_area.dev)
        
        out += ' ' + str(self.twoDr2.avg)
        out += ' ' + str(self.twoDr2.dev)
        
        out += ' ' + str(self.twoDtoG_area.avg)
        out += ' ' + str(self.twoDtoG_area.dev)
        
        out += ' ' + str(self.widthG.avg)
        out += ' ' + str(self.widthG.dev)
        
        out += ' ' + str(self.widthtwoD.avg)
        out += ' ' + str(self.widthtwoD.dev)
        
        out += ' ' + str(self.areaG.avg)
        out += ' ' + str(self.areaG.dev)
        
        out += ' ' + str(self.twoDtoG_height.avg)
        out += ' ' + str(self.twoDtoG_height.dev)
        
        print(out)    
        
  
    def printDtoG(self):
        print('D/G ratio by area: ', self.DtoG_area.getStr())
        print('D/G ratio by height: ', self.DtoG_height.getStr())
    
    def printtwoDtoG(self):
        if self.fitResults[0].best_values.get('twoD_center') != None:
            print('2D/G ratio by area: ', self.twoDtoG_area.avg, ' +/- ', self.twoDtoG_area.dev)
            print('2D/G ratio by height: ', self.twoDtoG_height.avg, ' +/- ', self.twoDtoG_height.dev)
    
    def histDtoGArea(self, save=None):
        
        if not save is None and save == True:
            saveName = '_DtoG-hist'
        else:
            saveName = None
        
        self.histFrom(
            self.DtoG_area.values,
            'D/G by area',
            bins=50,
            saveName=saveName
        )
    
    def histCentertwoD(self):
        self.histFrom(self.centertwoD.values, 'Center of 2D peak')
    
    def histtwoDtoGArea(self):
        if self.fitResults[0].best_values.get('twoD_center') != None:
            self.histFrom(
                self.twoDtoG_area.values,
                '2D/G by area',
                bins=50
            )
    
    def printPeakRatios(self, save=None):
        if self.datasetsNumber < 1:
            return
        self.printDtoG()
        self.histDtoGArea(save)
        
        self.printtwoDtoG()
        self.histtwoDtoGArea()
    
    def plotRes(self, resultNumber):
        if self.datasetsNumber < resultNumber:
            return
        plt.plot(self.x[resultNumber, :], self.fitResults[resultNumber].best_fit, 'r-')
        plt.plot(self.x[resultNumber, :], self.y[resultNumber, :])
        plt.xlabel(makePlotLabel(self.task, 'x'))
        plt.ylabel(makePlotLabel(self.task, 'y'))

    def plotDG(self, resultNumber):
        if self.datasetsNumber < resultNumber:
            return
        self.plotRes(resultNumber)
        plt.xlim(1250, 1650)
        plt.show()
        
    def plotFit(self, resultNumber):
        if self.datasetsNumber < 1:
            return
        
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        
        ax1.plot(self.x[resultNumber, :], self.fitResults[resultNumber].best_fit, 'r-')
        ax1.plot(self.x[resultNumber, :], self.y[resultNumber, :], 'ko', markersize=2)
        start, end = ax1.get_xlim()
        ax1.xaxis.set_ticks(np.arange(start, end, 100.0))
        
        ax2.plot(self.x[resultNumber, :], self.fitResults[resultNumber].best_fit, 'r-')
        ax2.plot(self.x[resultNumber, :], self.y[resultNumber, :], 'ko', markersize=2)
        start, end = ax2.get_xlim()
        ax2.xaxis.set_ticks(np.arange(start, end, 100.0))
        
        ax1.set_ylabel(makePlotLabel(self.task, 'y'))
        ax1.set_xlim([1250, 1650])
        ax2.set_xlim([2560, 2820])
        ax1.set_xlabel(makePlotLabel(self.task, 'x'))
        ax2.set_xlabel(makePlotLabel(self.task, 'x'))

        printmd('#### Example spectrum number: ' + str(resultNumber))
        plt.show()
    
    def plotAvgFit(self, saveName=None):
        if self.datasetsNumber < 1:
            return
        
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        
        ax1.plot(self.x[0, :], self.avgFitResult.best_fit, 'r-')
        ax1.plot(self.x[0, :], self.avgY, 'ko', markersize=2)
        start, end = ax1.get_xlim()
        ax1.xaxis.set_ticks(np.arange(start, end, 100.0))
        
        ax2.plot(self.x[0, :], self.avgFitResult.best_fit, 'r-')
        ax2.plot(self.x[0, :], self.avgY, 'ko', markersize=2)
        start, end = ax2.get_xlim()
        ax2.xaxis.set_ticks(np.arange(start, end, 100.0))
        
        ax1.set_ylabel(makePlotLabel(self.task, 'y'))
        ax1.set_xlim([1250, 1650])
        ax2.set_xlim([2560, 2820])
        ax1.set_xlabel(makePlotLabel(self.task, 'x'))
        ax2.set_xlabel(makePlotLabel(self.task, 'x'))
        
        plt.text(0.05, 0.95, 'D/G: \n' + self.derivedResults['DtoG_area'].getStr(decimalPlaces=2), fontsize=14, ha='left', va='center', transform=ax1.transAxes)
        plt.text(0.05, 0.95, '2D/G: \n' + self.derivedResults['twoDtoG_area'].getStr(decimalPlaces=2), fontsize=14, ha='left', va='center', transform=ax2.transAxes)
        
        if not saveName is None:
            plt.savefig(self.baseFilename + saveName + '.png', dpi=300)
            
        plt.show()
        
        
    def plot3D(self, fromX=None, toX=None, save=None):
        
        if len(self.fitResults) > 150:
            return
        
        if not fromX is None:
            fromX = self.task['xmin']
            toX = self.task['xmax']

        fig3D = plt.figure()
        ax = fig3D.add_subplot(111, projection='3d')
        
        
        for resultNumber, modResult in enumerate(self.fitResults):
            ax.plot(xs=self.x[resultNumber, :], ys=self.fitResults[resultNumber].best_fit, zs=resultNumber, zdir='y')
        
        if not fromX is None:
            ax.set_xlim([fromX, toX])
            
        ax.set_ylim([0, resultNumber])
        ax.set_zlim([0,  np.max(self.avgY)+ 2 * np.std(self.avgY)])
        ax.view_init(azim=250)
        ax.set_xlabel('\n\n' + makePlotLabel(self.task, 'x'))
        ax.set_zlabel('\n\n' + makePlotLabel(self.task, 'y'))
        ax.yaxis.set_ticklabels([])
        
        ax.text2D(0.3, 0.95, ('D/G: ' + self.derivedResults['DtoG_area'].getStr(decimalPlaces=2)), fontsize=14, ha='center', va='center', transform=ax.transAxes)
        ax.text2D(0.7, 0.99, ('2D/G: ' + self.derivedResults['twoDtoG_area'].getStr(decimalPlaces=2)), fontsize=14, ha='center', va='center', transform=ax.transAxes)
        
        if not save is None and save == True:
            plt.savefig(self.baseFilename + '_3Dplot_fittedData.png', dpi=300)
        
        plt.show()

    def plot2D(self, resultNumber, save=None):
        self.plotRes(resultNumber)
        plt.figtext(0.2, 0.8,
             'NSR: ' + str('{0:.2f}'.format(self.twoDmaxRes.values[resultNumber]))
        )
        
        plt.xlim(2560, 2800)
        
        if not save is None and save == True:
            plt.savefig('2DpeakRes' + str(resultNumber) + '.png')
        
        plt.show()
        
    def plot2Dresiduals(self, save=None):
    
        if self.datasetsNumber < 1:
            return
        if self.fitResults[0].params.get('twoD_center') is None:
            return
        
        #helper to get the running mean of a spectrum
        def running_mean(ydata, width):
            cumsum = np.cumsum(np.insert(ydata, 0, 0)) 
            return (cumsum[width:] - cumsum[:-width]) / width
        
        ax = plt.figure('2Dresiduals').add_subplot(111)
        ax.set_ylabel('Normalized squared residuals')
        ax.set_xlabel('Number of evaluated 2D data points')
        
        for i, modResult in enumerate(self.fitResults):
            
            #we use the range for the 2D peak defined in self.task['peaks']
            twoDresRange = self.weights['twoD'] == 1
            
            twoDres = self.fitResults[i].residual[twoDresRange,...]
            
            twoDres = twoDres**2 / (
                self.fitResults[i].params['twoD_height'])**2
            
            ax.plot(running_mean(twoDres, 5))

        plt.show()
        
        if not save is None and save == True:
            plt.savefig(self.baseFilename + '_2DpeakResiduals' + '.png') 
            
    
    def get2Dresiduals(self, evalRange=None, plot=None):
        if self.datasetsNumber < 1:
            return
        if self.fitResults[0].params.get('twoD_center') is None:
            return
        
        #helper to get the running mean of a spectrum
        def running_mean(ydata, width):
            cumsum = np.cumsum(np.insert(ydata, 0, 0)) 
            return (cumsum[width:] - cumsum[:-width]) / width
        
        for i, modResult in enumerate(self.fitResults):
            
            #we use the range for the 2D peak defined in self.task['peaks']
            twoDresRange = self.weights['twoD'] == 1
            
            twoDres = self.fitResults[i].residual[twoDresRange,...]
            ydata = self.y[i,:][twoDresRange,...]
            
            #the coefficient of determination is calculated from smoothed residuals
            #to reduce the impact of detector noise on the value
            #we are only interested in the symmetry of the peak
            r2 = 1 - np.var(running_mean(twoDres, 5)) / np.var(running_mean(ydata, 5))
            
            self.twoDr2.setValue(i, r2)
            
            twoDres = twoDres**2 / (
                self.fitResults[i].params['twoD_height'])**2
            
            self.twoDmaxRes.setValue(i, (max(twoDres)))
        
        self.twoDr2.setMask(0.0, 1)
        self.twoDmaxRes.setOutliers(5)
        self.twoDr2.setOutliers(5)
        
        self.twoDmaxRes.calcAvg()
        self.twoDr2.calcAvg()
        
        if not plot is None and plot == True: 
            self.plot2Dresiduals()
        
    def get2DresIntervals(self, thres):
        return (100 * (self.twoDmaxRes.values < thres).sum() /
                self.twoDmaxRes.values.count()
            )
    
    def get2Dr2Intervals(self, thres):
        return (100 * (self.twoDr2.values > thres).sum() /
                self.twoDr2.values.count()
            )
    
    def getDtoGIntervals(self, thres):
        return (100 * (self.DtoG_area.values < thres).sum() /
                self.DtoG_area.values.count()
            )
    
    def print2DresIntervals(self, thres):
        print('Symmetric 2D shapes (NSR < %s): %.2f' % (thres,
            self.get2DresIntervals(thres)),
              ' %'
        )
    
    def print2Dr2Intervals(self, thres):
        print('Symmetric 2D shapes (R2 > %s): %.2f' % (thres,
            self.get2Dr2Intervals(thres)),
              ' %'
        )
    
    def printDtoGIntervals(self, thres):
        print('D/G < %s): %.2f' % (thres,
            self.getDtoGIntervals(thres)),
              ' %'
        )
        
    def print2Dr2(self, save=None):   
        printmd('### 2D peak R2: ' + 
             self.twoDr2.getStr())
        
        
        if not save is None and save == True:
            saveName = '_R2-hist'
        else:
            saveName = None
            
        self.histFrom(
            self.twoDr2.values,
            'Coefficient of determination R2',
            bins=np.linspace(0.9, 1, 51),
            saveName=saveName
        )
    
    def print2Dresiduals(self, save=None): 
        printmd('### Maximum normalized residual of 2D peak fit: ' +
             self.twoDmaxRes.getStr())
        
        if not save is None and save == True:
            saveName = '_NSR-hist'
        else:
            saveName = None
        
        self.histFrom(
            self.twoDmaxRes.values,
            'Normalized squared residual',
            bins=np.linspace(0, 0.1, 51),
            saveName=saveName
        )
    
    def plot2DOutliers(self, outlierNumber, save=None):
        if self.datasetsNumber < 1:
            return
        
        outliersFig = plt.figure()
        #outliersFig.subplots_adjust(left=-0.1, right=0.9, top=1.0, bottom=-0.3, hspace=0.2, wspace=.001)
        outliersFig.set_figheight(outlierNumber * 6)
        
        def makeSubPlot(fromIndices, col):
            for i, index in enumerate(fromIndices):
                if i < outlierNumber:
                    outlier = outliersFig.add_subplot(outlierNumber, 2, 2 * i + col)
                    outlier.plot(self.x[index, :], self.fitResults[index].best_fit, 'r-', linewidth=2.0)
                    outlier.plot(self.x[index, :], self.y[index, :], linewidth=2.0)
                    if i >=  outlierNumber - 1:   
                        outlier.set_xlabel(makePlotLabel(self.task, 'x'))
                    if col == 1:
                        outlier.set_ylabel(makePlotLabel(self.task, 'y'))
                    #plt.tick_params(
                    #    axis='y',
                    #    which='both',
                    #    left='off',
                    #    right='off',
                    #    labelleft='off'
                    #)
                    outlier.set_title(
                        'Spec. ' + str(index) +
                        ', R2: %.3f' % self.twoDr2.values[index]
                    )
                    outlier.set_xlim(2560, 2800)

        
        makeSubPlot(self.twoDr2.outliersMin, 1)
        makeSubPlot(self.twoDr2.outliersMax, 2)
        
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
       
        if not save is None and save == True:
            outliersFig.savefig(self.baseFilename + '_2DpeakR2.png', dpi=300)
    
        plt.show()
        
    def plotDtoGOutliers(self, outlierNumber, save=None):
        if self.datasetsNumber < 1:
            return
        
        outliersFig = plt.figure()
        #outliersFig.subplots_adjust(left=-0.1, right=0.9, top=1.0, bottom=-0.3, hspace=0.2, wspace=.001)
        outliersFig.set_figheight(outlierNumber * 6)
        
        def makeSubPlot(fromIndices, col):
            for i, index in enumerate(fromIndices):
                if i < outlierNumber:
                    outlier = outliersFig.add_subplot(outlierNumber, 2, 2 * i + col)
                    outlier.plot(self.x[index, :], self.fitResults[index].best_fit, 'r-')
                    outlier.plot(self.x[index, :], self.y[index, :])
                    if i >=  outlierNumber - 1:   
                        outlier.set_xlabel(makePlotLabel(self.task, 'x'))
                    if col == 1:
                        outlier.set_ylabel(makePlotLabel(self.task, 'y'))
                    #plt.tick_params(
                    #    axis='y',
                    #    which='both',
                    #    left='off',
                    #    right='off',
                    #    labelleft='off'
                    #)
                    outlier.set_title(
                        'Spec. ' + str(index) +
                        ', D/G: %.2f' % self.DtoG_area.values[index]
                    )
                    outlier.set_xlim(1250, 1750)

        
        makeSubPlot(self.DtoG_area.outliersMin, 1)
        makeSubPlot(self.DtoG_area.outliersMax, 2)
        
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
       
        if not save is None and save == True:
            outliersFig.savefig(self.baseFilename + '_DtoG.png', dpi=300)
    
        plt.show()
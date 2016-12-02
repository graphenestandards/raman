import numpy as np
from lmfit.models import PolynomialModel, PseudoVoigtModel
from lmfit.model import ModelResult
from lmfit import Parameters, report_fit

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import savefig
import time
from copy import deepcopy

import sys

import csv

from IPython.display import HTML, display, Markdown
#we want to use markdown in the output of cells
def printmd(string):
    display(Markdown(string))

import tkinter as tk
from tkinter import filedialog    
    
    
class Dataset:
    def __init__(self, task=None):
        self.weights = {}
        self.task = task
        
        if task.get('filename') is None:
            task['filename'] = self.getFilename(initialDir=task.get('initialDir'))
        
        self.data, self.numberOfDatasets = self.readDatasets(self.task)
        self.datasetX, self.datasetY, self.posX, self.posY = self.prepareDatasets(self.data, self.task)
        self.avgY = np.average(self.datasetY, axis=0)
        
        self.setWeights(self.task.get('peaks'))
        
    def getFilename(self, initialDir=None):
        
        if initialDir is None:
            initialDir = './'
        filename = filedialog.askopenfilename(
            initialdir=initialDir,
            filetypes=[('Raman text files', '*.txt')]
        )
        
        root = tk.Tk()
        root.withdraw()
        
        return filename
    
    def readDatasets(self, task):
        
        print('Reading ', task['filename'], flush=True)

        usecols = [
            task['ramanshift_column'],
            task['ramanintensity_column']
        ]

        if not task.get('posx_column') is None and not task.get('posy_column') is None:
            usecols.append(task['posx_column'])
            usecols.append(task['posy_column'])

        #if bool(task.get('posx_') and pos
        data = np.loadtxt(task['filename'], usecols=usecols)

        print('Data ', task['filename'], ' read', flush=True)

        #if data are in descending order, reverse
        if data[1,0] < data[0,0]:
            data[:] = data[::-1]

        print('Data reversed', flush=True)

        #detect a new dataset if x value smaller than preceding xval
        number_of_datasets = 1
        xval = data[0,0]
        for nextx in data[:,0]:
            if nextx < xval:
                number_of_datasets += 1
            xval = nextx

        print(str(number_of_datasets), ' datasets detected', flush=True)

        return (data, number_of_datasets)


    def prepareDatasets(self, data, task):
        """Build x, y and position arrays
        
        Cuts the dataset to the range specified in the task.
        If necessary, brings the data in ascending-x order
        Args:
            data (numpy.array): Data from readDatasets
            task (dict): Task object defining xmin and xmax
        Returns:
            tuple: tuple of 2D numpy arrays for x, y, [posX, and posY]
        """
        
        #xmax and xmin are optional in the task dict
        if (not task.get('xmax') is None and
            not task.get('xmin') is None):

            mask = np.ones(len(data), dtype=bool)

            mask = np.where(
                ((data[:,0] > task['xmin']) &
                (data[:,0] < task['xmax'])),
                True,
                False
            )
            data = data[mask,...]
            print('Data cut: ', task['xmin'], task['xmax'], flush=True)

        task['datasetsLoaded'] = self.numberOfDatasets
        
        x = np.reshape(data[:,0], (task['datasetsLoaded'], -1))
        y = np.reshape(data[:,1], (task['datasetsLoaded'], -1))

        print('Data reshaped', flush=True)

        if not task.get('posx_column') is None and not task.get('posy_column') is None:
            posx = np.reshape(data[:,2], (task['datasetsLoaded'], -1))
            posy = np.reshape(data[:,3], (task['datasetsLoaded'], -1))
            #posx = posx[np.array([True]),...]
            #posy = posy[np.array([True]),...]
        else:
            posx = None
            posy = None

        return (x, y, posx, posy)
    
    def updateAvgSpec(self):
        self.avgY = np.average(self.datasetY, axis=0)
    
    def setWeights(self, peaks):
        """Set weights arrays
        Here we generate weights arrays, one for each peak and one for the baseline.
        For each peak, the weight is zero except in the range defined in the peak dictionary.
        The baseline has weight 1 everywhere except in the peak ranges, where it is set to zero.
        Like that, the baseline fit ignores the peak ranges.
        
        Args:
            peaks (dict): a list of peaks
        """
        x = self.datasetX[0,:]

        weightsBaseline = np.ones_like(x)

        for peak in peaks:
            weightsPeak = np.where(
                ((x > peak['range']['min']) &
                (x < peak['range']['max'])),
                1.0,
                0.0
            )
            weightsBaseline = np.where(
                ((x > peak['range']['min']) &
                (x < peak['range']['max'])),
                0.0,
                weightsBaseline
            )

            self.weights[peak['prefix']] = weightsPeak

        self.weights['baseline'] = weightsBaseline
        self.weights['peaks'] = 1 - weightsBaseline
        self.weights['included'] = np.ones_like(self.datasetX[:,0])


        print('Weights arrays initialized', flush=True)
        
    
    def subtractBaseline(self, exampleSpec=None, baselinePart=None, mask=None):
        """Subtracts a partially defined polynomial baseline
        
        Args:
            exampleSpec (int): index of an example spectrum to be plotted.
                If this is None, no output will be plotted.
            baselinePart (dict): includes 'polynomialOrder'
            mask (numpy.array): a boolean array of the shape like datasetX.
                Only the part where True will be baseline corrected.
        """
        if any(dim < 1 for dim in np.shape(self.datasetX)):
            return

        cutDatasetX = self.datasetX[:,mask,...]
        cutDatasetY = self.datasetY[:,mask,...]
        cutBaselineweights = self.weights['baseline'][mask,...]

        if not baselinePart is None:
            baselineOrder = baselinePart['polynomialOrder']
            minSnr = baselinePart.get('snr')
        else:
            baselineOrder = self.task.get('baselineOrder')
            if baselineOrder is None or baselineOrder < 1:
                baselineOrder = 1
            checkSnr = True
            minSnr = self.task.get('snr')

        if minSnr is None:
            minSnr = 0

        if baselineOrder is None or baselineOrder < 1:
            correctData = False
            baselineOrder = 6
        else:
            correctData = True

        snrFilter = np.ones_like(cutDatasetX[:,0], dtype=bool)

        snrs = np.zeros_like(cutDatasetX[:,0])


        parameters = Parameters()
        mod = PolynomialModel(baselineOrder, prefix='background_')
        parameters.update(mod.guess(cutDatasetY[0,:], x=cutDatasetX[0,:]))


        for i, dataX in enumerate(cutDatasetX):

            try:
                modResult = mod.fit(cutDatasetY[i,:], parameters, x=cutDatasetX[i,:], weights=cutBaselineweights)

                baseline = mod.eval(modResult.params, x=cutDatasetX[i,:])

                if exampleSpec is not None and i == exampleSpec:
                    exampleBaselineData = mod.eval(modResult.params, x=cutDatasetX[i,:])
                    exampleResiduals = modResult.residual

                if correctData:
                    self.datasetY[i,mask,...] = cutDatasetY[i,:] - baseline

                #calc signal to noise - note that residuals exclude the peak area
                snr = np.var(cutDatasetY[i,:])**2 / np.var(modResult.residual)**2

                if snr != snr:
                    snr = 0

                snrs[i] = snr

                if snr < minSnr:
                    snrFilter[i] = False                        

            except Exception as err:
                print(err)
                snrFilter[i] = False
            self.weights['included'] = np.logical_and(self.weights['included'], snrFilter)

        #plot output only if exampleSpec is set
        if not exampleSpec is None:

            printmd('## Example baseline')
            baselineFig = plt.figure()
            baselineFig.subplots_adjust(left=0.1, right=0.9, top=1.0, bottom=-0.3)
            baselineFig.set_figheight(18)

            exampleBaseline = baselineFig.add_subplot(421)
            exampleBaseline.set_title('Example baseline Nr. %d' % exampleSpec)
            exampleResidual = baselineFig.add_subplot(422)
            exampleResidual.set_title('Example residual Nr. %d' % exampleSpec)

            exampleBaseline.plot(cutDatasetX[exampleSpec, :], cutDatasetY[exampleSpec, :], 'r--')
            exampleBaseline.plot(cutDatasetX[exampleSpec, :], exampleBaselineData)
            #exampleBaseline.set_xlabel('Raman shift (cm-1)')
            exampleBaseline.set_ylabel('Intensity (arb. units)')

            exampleResidual.plot(cutDatasetX[exampleSpec, :], exampleResiduals)
            #exampleResidual.set_xlabel('Raman shift (cm-1)')
            exampleResidual.set_ylabel('Intensity (arb. units)')


            includedNumber = np.sum(snrFilter)
            excludedNumber = len(snrFilter) - includedNumber

            #if minSnr > 0 and (0 < np.sum(snrFilter) < len(snrFilter)):
            if minSnr > 0 and excludedNumber > 0:

                exampleExcluded = baselineFig.add_subplot(423)
                exampleExcluded.set_title('Examples from %d excluded spectra' % excludedNumber)

                exampleIncluded = baselineFig.add_subplot(424)
                exampleIncluded.set_title('Example included spectra')

                exclInclInds = np.argsort(snrs)

                maxPlots = 5
                includedPlotted = 0
                excludedPlotted = 0
                firstExcludedSpec = 0

                for i, ind in enumerate(exclInclInds):
                    if snrFilter[ind] == True:
                        atSnr = i
                        firstExcludedSpec = i - min(excludedNumber, maxPlots)
                        firstExcludedSpec = max(0, firstExcludedSpec)
                        firstIncludedSpec = i + min(maxPlots, includedNumber)
                        firstIncludedSpec = max(firstIncludedSpec, len(self.datasetX[:,0]) - 1)
                        break

                for ind in exclInclInds[firstExcludedSpec:atSnr]:
                    exampleExcluded.plot(self.datasetX[ind,mask,...], self.datasetY[ind,mask,...])

                for ind in exclInclInds[atSnr:firstIncludedSpec]:
                    exampleIncluded.plot(self.datasetX[ind,mask,...], self.datasetY[ind,mask,...])

                exampleExcluded.set_ylabel('Intensity (arb. units)')
                exampleIncluded.set_xlabel('Raman shift (cm-1)')
                exampleIncluded.set_ylabel('Intensity (arb. units)')

                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
                plt.show()

                plt.hist(snrs)
                plt.xlabel('Signal to noise ratio')
                plt.ylabel('#')

            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.show()
            printmd('## Example for corrected spectrum')
            plt.title('Example baseline corrected spectrum Nr. %d' % exampleSpec)
            plt.plot(self.datasetX[exampleSpec, :], self.datasetY[exampleSpec, :])
            plt.xlabel('Raman shift (cm-1)')
            plt.ylabel('Intensity (arb. units)')
            plt.show()


    def filterDatasetBySNR(self, exampleSpec=None):
        """Filters out data with too low SNR
            Applies the filter of self.weights['included'] to the dataset.
            
            Args:
                exampleSpec (int): index of an example spectrum to be plotted
        """
        
        mask = self.weights.get('included')

        if (not (mask is None) and
            len(mask) - np.sum(mask) > 0):

            print('%d spectra were excluded from %d datasets' %
                  (len(mask) - np.sum(mask), self.task['datasetsLoaded']))

            x = self.datasetX[mask,...]
            y = self.datasetY[mask,...]
        else:
            print('%d spectra were excluded from %d datasets' % (0, self.task['datasetsLoaded']))

            x = self.datasetX
            y = self.datasetY


        if len(x[:, 0]) < self.task['minimumStatisctics']:
            raise UserWarning((
                'The mimimum size of %d spectra'  % self.task['minimumStatisctics'] +
                ' with a high enough signal-to-noise' +
                ' level has not been reached.' +
                ' You can change the \'minimumStatisctics\' in the task,' +
                ' the evaluation will then not any more correspond to the standard.'))

        else:
            print('%d datasets left for fitting' % len(x[:, 0]))
            self.task['datasetsFitted'] = len(x[:, 0])
            
        self.datasetX = x
        self.datasetY = y
        
        self.updateAvgSpec()
        

    #instead of a single baseline, a partially defined polynomial baseline can be used
    def subtractMultiBaseline(self, baselineParts=None, exampleSpec=None):

        lastXBoundary = self.datasetX[0,0]

        for i, baselinePart in enumerate(baselineParts):

            baselineMask = np.where(((self.datasetX[0,:] < baselinePart.get('untilX')) &
                (self.datasetX[0,:] >= lastXBoundary)),
                True,
                False
            )

            self.subtractBaseline(
                exampleSpec=exampleSpec,
                baselinePart=baselinePart,
                mask=baselineMask)

            lastXBoundary = baselinePart.get('untilX')
        
        self.updateAvgSpec()

        
        
        
        
class MultiPseudoVoigtModel:
    #static paramNames
    paramNames = ['center', 'sigma', 'fwhm', 'amplitude', 'fraction', 'height']
    
    def __init__(self, dataset, weights=None):
        self.peaks = dataset.task.get('peaks')
        self.peakNames = []
        self.paramNames = MultiPseudoVoigtModel.paramNames
        
        self.datasetX = dataset.datasetX
        self.datasetY = dataset.datasetY
        self.avgY = dataset.avgY
        
        self.startTime = time.time(),
        self.weights = weights
        
        self.fitResults = np.zeros_like(self.datasetX[:,0], dtype=object)
        self.parameters = Parameters()
        if len(self.datasetX) > 0:
            self.makeModel()
        
    def makePeak(self, peak):
        
        prefix = peak['prefix'] + '_'
        
        if peak['vary'] == True:
            self.peakNames.append(peak['prefix'])
        
        peakMod = PseudoVoigtModel(prefix=prefix)
        
        #the lmfit PseudoVoigt model does not have a height
        #add this as fixed parameter
        self.parameters.add(
            prefix + 'height'
        )
           
        self.parameters.update(
            peakMod.guess(self.datasetY[0,:], x=self.datasetX[0,:]))
        
        for peakParameter in MultiPseudoVoigtModel.paramNames: 
            self.parameters[prefix + peakParameter].set(
                peak.get(peakParameter, {}).get('init'),
                min=peak.get(peakParameter, {}).get('min'),
                max=peak.get(peakParameter, {}).get('max'),
                vary=peak.get(peakParameter, {}).get('vary')
            )
        
        #override default in lmfit model
        self.parameters[prefix + 'fraction'].set(
            peak.get('fraction', {}).get('init'),
            vary=True,
            expr=None
        )
        
        #override default in lmfit model
        self.parameters[prefix + 'height'].set(
            100.0,
            vary=False,
            expr='%samplitude / %sfwhm' % (prefix, prefix)
        )
        
        self.parameters[prefix + 'fwhm'].set(
            100.0,
            vary=False,
            expr='%ssigma * 2.0' % (prefix)
        )
        
        
        return peakMod
    
    
    def makeModel(self):
        peaksIncluded = 0
        
        for i, peak in enumerate(self.peaks):
            peakMod = self.makePeak(peak)
            
            if peak['vary'] == True:
                if peaksIncluded == 0:
                    self.model = peakMod
                    peaksIncluded += 1
                else:
                    self.model = self.model + peakMod
                    peaksIncluded += 1
    
    
    def runFit(self, maxNumber=None):
        self.startTime = time.time()
        print('--- Fitting datasets started ',
              time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()),
              flush=True)
        try:
            self.avgFitResult = self.model.fit(
                self.avgY,
                self.parameters,
                x=self.datasetX[0,:],
                method='leastsq',
                weights=self.weights
            )
        except Exception as err:
            print(err)
            
        for i, ydata in enumerate(self.datasetY):
 
            if not maxNumber is None and i > maxNumber:
                break

            try:
                modResult = self.model.fit(
                    self.datasetY[i,:],
                    self.parameters,
                    x=self.datasetX[i,:],
                    method='leastsq',
                    weights=self.weights
                )

                self.fitResults[i] = modResult
                
            except Exception as err:
                print(err)
                #self.excludedDatasets.append(i)

        print("--- Fit completed in %s seconds ---" % (time.time() - self.startTime))
        
        
class FitResultParameter:
    def __init__(self, fitResults, paramName=None):
        self._values = fitResults
        self.paramName = paramName
        #self.maskedValues = self.values
        self._avg = 0
        self._dev = 0
        self.outliersMin = np.ones_like(fitResults, dtype=int)
        self.outliersMax = np.ones_like(fitResults, dtype=int)
        self.calcAvg()
    @property
    def avg(self):
        return self._avg
    @property
    def dev(self):
        return self._dev
    @property
    def values(self):
        return self._values
    @values.setter
    def values(self, values):
        self._values = values
        self.calcAvg()
    def setValue(self, index, value):
        self._values[index] = value
    def calcAvg(self):
        self._avg = np.ma.average(self._values)
        self._dev = np.ma.std(self._values)
    def getStr(self):
        return '%.3f +/- %.3f' % (self._avg, self._dev)
    def setMask(self, lower, upper):
        self.values = np.ma.masked_outside(self._values, lower, upper)
        #self.values.fill_value = np.nan
    def setOutliers(self, howMany):
        if howMany > len(self._values) + 1:
            howMany = len(self._values) - 1
        self.outliersMin = np.ma.masked_array.argpartition(
            np.ma.filled(self.values, self.avg), howMany)[:(howMany)]
        
        self.outliersMax = np.ma.masked_array.argpartition(
            np.ma.filled(self._values, self._avg), -howMany)[-(howMany):]


class MultiPeakFitResults:
    def __init__(self, fitResults, peakNames, paramNames):
        self.fitResults = fitResults
        
        self.peakNames = peakNames
        self.paramNames = paramNames
        self.peakParamNames = {}
        
        self.peakResults = np.empty(
            (len(self.peakNames), len(self.paramNames)), dtype=object)
              
        for peakNo, peakName in enumerate(self.peakNames):
            for paramNo, paramName in enumerate(self.paramNames):
                fitParamResults = np.ones(len(fitResults))
                peakParamName = peakName + '_' + paramName
                self.peakParamNames[peakParamName] = (peakNo, paramNo)
                for resultNo, modResult in enumerate(fitResults):
                    fitParamResults[resultNo] = modResult.params.get(peakParamName)
                        
                self.peakResults[peakNo][paramNo] = FitResultParameter(
                    fitParamResults, paramName=peakParamName)
        
    def getReferenceByName(self, peakParamName):
        return self.peakResults[self.peakParamNames[peakParamName]]
        
    def getStr(self):
        for resNo, res in np.ndenumerate(self.peakResults):
            print(res.paramName, res.getStr())
    
    def getHTML(self):
        return '<table><tr>{}</tr></table>'.format(
            '<tr><td></td>' + (
                '<td>{}</td>'.format(
                    '</td><td>'.join(
                        paramName for paramName in self.paramNames))) +
            '</tr>' +
            '</tr><tr>'.join(
                '<td>' +
                self.peakNames[rowNo] +
                '</td>' +
                '<td>{}</td>'.format(
                    '</td><td>'.join(
                        col.getStr() for col in row)) for rowNo, row in enumerate(self.peakResults))
            )
        
    def printHTML(self):
        display(HTML(self.getHTML()))
            
        
class GrapheneModelResults:
    def __init__(self, fittedModel, dataset):
        self.fitResults = fittedModel.fitResults
        self.avgFitResult = fittedModel.avgFitResult
        self.avgY = dataset.avgY
        self.x = dataset.datasetX
        self.y = dataset.datasetY
        
        self.posX = dataset.posX
        self.posY = dataset.posY
        self.weights = dataset.weights
        
        self.datasetsNumber = len(self.x)
        
        self.task = dataset.task
        self.baseFilename = self.task['filename'].replace('.txt', '')
        
        self.DtoG_area = FitResultParameter(np.ones_like(self.x[:,0]))
        self.DtoG_height = FitResultParameter(np.ones_like(self.x[:,0]))
        self.twoDtoG_area = FitResultParameter(np.ones_like(self.x[:,0]))
        self.twoDtoG_height = FitResultParameter(np.ones_like(self.x[:,0]))
       
        self.twoDmaxRes = FitResultParameter(np.ones_like(self.x[:,0]))
        self.twoDr2 = FitResultParameter(np.ones_like(self.x[:,0]))
        
        self.peakResults = MultiPeakFitResults(
            self.fitResults, fittedModel.peakNames, fittedModel.paramNames)
        
        self.widthD = self.peakResults.getReferenceByName('D_fwhm')
        self.widthG = self.peakResults.getReferenceByName('G_fwhm')
        self.widthtwoD = self.peakResults.getReferenceByName('twoD_fwhm')
        
        
        self.centerD = self.peakResults.getReferenceByName('D_center')
        self.centerG = self.peakResults.getReferenceByName('G_center')
        self.centertwoD = self.peakResults.getReferenceByName('twoD_center')
        
        self.areaG = self.peakResults.getReferenceByName('G_amplitude')
        
        self.calcPeakRatios()
    
    def calcPeakRatios(self):
        if self.datasetsNumber < 1:
            return
        if self.fitResults[0].params.get('G_amplitude') is None:
            return
        
        self.DtoG_area.values = (
            self.peakResults.getReferenceByName('D_amplitude').values /
                self.peakResults.getReferenceByName('G_amplitude').values)
        
        self.DtoG_height.values = (
            self.peakResults.getReferenceByName('D_height').values /
                self.peakResults.getReferenceByName('G_height').values)
        
        self.twoDtoG_area.values = (
            self.peakResults.getReferenceByName('twoD_amplitude').values /
                self.peakResults.getReferenceByName('G_amplitude').values)
        
        self.twoDtoG_height.values = (
            self.peakResults.getReferenceByName('twoD_height').values /
                self.peakResults.getReferenceByName('G_height').values)
           
        self.DtoG_area.setMask(0.01, 5)
        self.DtoG_height.setMask(0.01, 5)
        self.DtoG_area.calcAvg()
        self.DtoG_area.setOutliers(5)
        self.DtoG_height.calcAvg()
        self.DtoG_height.setOutliers(5)
        self.twoDtoG_area.calcAvg()
        self.twoDtoG_height.calcAvg()
        
    def printResultTable(self):
        self.peakResults.printHTML()
    
    def writeResultTable(self, filenameAd):
        filename = self.baseFilename + '_' + filenameAd + '.csv'
        delimiter = ';'
        delimiter = delimiter
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=delimiter,
                                    quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['sep=;'])
            row = ['Peak']
            for paramName in self.peakResults.paramNames:
                row.append(paramName)
                row.append('+/-')
            writer.writerow(row)
            for rowNo, peakRow in enumerate(self.peakResults.peakResults):
                row = [self.peakResults.peakNames[rowNo]]
                for param in peakRow:
                    row.append(str(param.avg))
                    row.append(str(param.dev))
                writer.writerow(row)
        
            print('Peak results saved as: ', filename)
            
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
    
    def printWidths(self):
        if self.fitResults[0].best_values.get('D_center') != None:
            print('D width: ', self.widthD.avg, ' +/- ', self.widthD.dev)
        if self.fitResults[0].best_values.get('G_center') != None:    
            print('G width: ', self.widthG.avg, ' +/- ', self.widthG.dev)
        
        if self.fitResults[0].best_values.get('twoD_center') != None:
            print('2D width: ', self.widthtwoD.avg, ' +/- ', self.widthtwoD.dev)
        
    def plotWidths(self):
        self.histFrom(self.widthD.values, 'D FWHM (cm-1)')
        self.histFrom(self.widthG.values, 'G FWHM (cm-1)')
        self.histFrom(self.widthtwoD.values, '2D FWHM (cm-1)')
    
    def histFrom(self, values, xlabel, saveName=None, bins=None):
        if not bins is None:
            plt.hist(values, bins=bins)
        else:
            plt.hist(values)
        plt.xlabel(xlabel)
        plt.ylabel('#')
        if not saveName is None:
            plt.savefig(self.baseFilename + saveName + '.png', dpi=300)
        plt.show()
    
    def histDtoGArea(self, save=None):
        
        if not save is None and save == True:
            saveName = '_DtoG-hist'
        else:
            saveName = None
        
        self.histFrom(
            self.DtoG_area.values,
            'D/G',
            bins=np.linspace(0, 2, 41),
            saveName=saveName
        )
    
    def histCentertwoD(self):
        self.histFrom(self.centertwoD.values, 'Center of 2D peak')
    
    def histtwoDtoGArea(self):
        if self.fitResults[0].best_values.get('twoD_center') != None:
            self.histFrom(self.twoDtoG_area.values, '2D/G')
    
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
        plt.xlabel('Raman shift (cm-1)')
        plt.ylabel('Intensity (arb. units)')

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
        
        ax1.set_ylabel('Intensity (arb. units)')
        ax1.set_xlim([1250, 1650])
        ax2.set_xlim([2550, 2820])
        ax1.set_xlabel('Raman shift (cm-1)')
        ax2.set_xlabel('Raman shift (cm-1)')
        plt.show()
    
    def plotAvgFit(self):
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
        
        ax1.set_ylabel('Intensity (arb. units)')
        ax1.set_xlim([1250, 1650])
        ax2.set_xlim([2550, 2820])
        ax1.set_xlabel('Raman shift (cm-1)')
        ax2.set_xlabel('Raman shift (cm-1)')
        plt.show()

    def plot2D(self, resultNumber, save=None):
        self.plotRes(resultNumber)
        plt.figtext(0.2, 0.8,
             'NSR: ' + str('{0:.2f}'.format(self.twoDmaxRes.values[resultNumber]))
        )
        
        plt.xlim(2550, 2800)
        
        if not save is None and save == True:
            plt.savefig('2DpeakRes' + str(resultNumber) + '.png')
        
        plt.show()
        
    
    def plot3D(self, save=None):
        
        if len(self.fitResults) > 150:
            return
        
        fig3D = plt.figure()
        ax = fig3D.add_subplot(111, projection='3d')
        
        
        
        for resultNumber, modResult in enumerate(self.fitResults):
            ax.plot(xs=self.x[resultNumber, :], ys=self.fitResults[resultNumber].best_fit, zs=resultNumber, zdir='y')
        
        
        ax.set_xlim([1250, 2850])
        ax.set_ylim([0, resultNumber])
        ax.set_zlim([0,  np.max(self.avgY)+ 2 * np.std(self.avgY)])
        ax.view_init(azim=250)
        ax.xaxis._axinfo['label']['space_factor'] = 6.8
        ax.zaxis._axinfo['label']['space_factor'] = 6.8
        ax.set_xlabel('Raman shift (cm-1)')
        ax.set_zlabel('Intensity')
        
        if not save is None and save == True:
            plt.savefig('3Dplot_fittedSpectra.png')
            
        plt.show()
        
    
    
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
            
            if not plot is None and plot == True: 
                plt.plot(running_mean(twoDres, 5))
        
        self.twoDr2.setMask(0.0, 1)
        self.twoDmaxRes.setOutliers(5)
        self.twoDr2.setOutliers(5)
        
        if not plot is None and plot == True:
            plt.ylabel('Normalized squared residuals')
            plt.xlabel('Number of evaluated 2D data points')
            plt.show()
            
        self.twoDmaxRes.calcAvg()
        self.twoDr2.calcAvg()
		
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
                        outlier.set_xlabel('Raman shift (cm-1)')
                    if col == 1:
                        outlier.set_ylabel('Intensity (arb. units)')
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
                    outlier.set_xlim(2550, 2800)

        
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
                        outlier.set_xlabel('Raman shift (cm-1)')
                    if col == 1:
                        outlier.set_ylabel('Intensity (arb. units)')
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
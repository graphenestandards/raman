from . import *
    
    
class Dataset:
    def __init__(self, task=None, maxNumber=None):
        self.weights = {}
        self.task = task
        
        if task.get('filename') is None:
            task['filename'] = self.getFilename(initialDir=task.get('initialDir'))
        
        self.data, self.numberOfDatasets = self.readDatasets(self.task, maxNumber)
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
    
    def readDatasets(self, task, maxNumber=None):
        
        print('Reading ', task['filename'], flush=True)

        usecols = [
            task['x_column'],
            task['y_column']
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

        for indX, nextx in enumerate(data[:,0]):
            if nextx < xval:
                if not maxNumber is None and number_of_datasets >= maxNumber:
                    data = data[0:indX-1,:]
                    break
                else:
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
            exampleBaseline.set_ylabel(self.task.get('ylabel'))

            exampleResidual.plot(cutDatasetX[exampleSpec, :], exampleResiduals)
            exampleResidual.set_ylabel(self.task.get('ylabel'))


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

                exampleExcluded.set_ylabel(self.task.get('ylabel'))
                exampleIncluded.set_xlabel(self.task.get('xlabel'))
                exampleIncluded.set_ylabel(self.task.get('ylabel'))

                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
                plt.show()

                plt.hist(snrs)
                plt.xlabel('Signal to noise ratio')
                plt.ylabel('#')

            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
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
        printmd('## Example for corrected spectrum')
        
        plt.title('Example baseline corrected spectrum Nr. %d' % exampleSpec)
        plt.plot(self.datasetX[exampleSpec, :], self.datasetY[exampleSpec, :])
        plt.xlabel(self.task.get('xlabel'))
        plt.ylabel(self.task.get('ylabel'))
        plt.show()

        
        
        
        
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

        if not maxNumber is None and maxNumber < len(self.datasetX[:,0]):
            self.fitResults = np.zeros(maxNumber, dtype=object)
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
 
            if not maxNumber is None and i >= maxNumber:
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
            np.ma.filled(self._values, self._avg), howMany)[:(howMany)]
        
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
    
    def getHTML(self):
        return ('<div>Results from ' + str(len(self.fitResults)) + ' datasets:' +
            '<table><tr>{}</tr></table>'.format(
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
            ))

        
class MultiPeakModelResults:
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

        self.peakResults = MultiPeakFitResults(
            self.fitResults, fittedModel.peakNames, fittedModel.paramNames)

    def calcRatio(self, param1, param2, min=None, max=None, outliers=None):
        
        fitResultParameter = FitResultParameter(np.ones_like(self.x[:,0]))

        if self.datasetsNumber < 1:
            return fitResultParamter
        if self.fitResults[0].params.get(param2) is None:
            return fitResultParamter
        if self.fitResults[0].params.get(param2) is None:
            return fitResultParamter

        fitResultParameter.values = (
            self.peakResults.getReferenceByName(param1).values /
                self.peakResults.getReferenceByName(param2).values)

        if not min is None:
            fitResultParameter.setMask(min, max)
        if not outliers is None:
            fitResultParameter.setOutliers(outliers)

        fitResultParameter.calcAvg()

        return fitResultParameter
   
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
            ax.set_xlim([1250, 2850])
            
        ax.set_ylim([0, resultNumber])
        ax.set_zlim([0,  np.max(self.avgY)+ 2 * np.std(self.avgY)])
        ax.view_init(azim=250)
        ax.xaxis._axinfo['label']['space_factor'] = 6.8
        ax.zaxis._axinfo['label']['space_factor'] = 6.8
        ax.set_xlabel(self.task.get('xlabel'))
        ax.set_zlabel(self.task.get('ylabel'))
        
        if not save is None and save == True:
            plt.savefig('3Dplot_fittedData.png')
            
        plt.show()

    def printWidths(self):
        self.printParamResults('fwhm')
    
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

    def plotPeakParamHist(self, peakName, paramName, xunit=None, saveName=None, bins=None):
        resultName = peakName + '_' + paramName
        values = self.peakResults.getReferenceByName(resultName).values
        xlabel = resultName
        if not xunit is None:
            xlabel += ' (' + xunit + ')'

        self.histFrom(values, xlabel, saveName=saveName, bins=bins)

    def plotWidths(self):
        for peakName in self.peakResults.peakNames:
            self.plotPeakParamHist(peakName, 'fwhm', xunit='cm-1')

    def writeResultTable(self, filenameAd):
        filename = self.baseFilename + '_' + filenameAd + '.csv'
        delimiter = ';'
        
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

    def printResultTable(self):
        display(HTML(self.peakResults.getHTML()))

    def printParamResults(self, paramName):
        for peakName in self.peakResults.peakNames:
            print(peakName + ' ' + paramName + ': ' +
                self.peakResults.getReferenceByName(
                    peakName + '_' + paramName).getStr())

            
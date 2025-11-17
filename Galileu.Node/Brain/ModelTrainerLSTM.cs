using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using Galileu.Node.Core;
using Galileu.Node.Gpu;
using Galileu.Node.Interfaces;
using Galileu.Node.Services;

namespace Galileu.Node.Brain
{
    public class ModelTrainerLSTM
    {
        private readonly IMathEngine _mathEngine;
        private readonly Stopwatch _stopwatch = new Stopwatch();
        private readonly Process _currentProcess;
        private readonly ISearchService _searchService;
        private readonly GpuSyncGuard? _syncGuard;
        private long _peakMemoryUsageMB = 0;
        private readonly string logPath = Path.Combine(Environment.CurrentDirectory, "Dayson", "training_log.txt");

        private const long MEMORY_TRIM_THRESHOLD_MB = 2000;
        private long _lastTrimMemory = 0;

        public ModelTrainerLSTM(IMathEngine mathEngine)
        {
            _mathEngine = mathEngine ?? throw new ArgumentNullException(nameof(mathEngine));
            _currentProcess = Process.GetCurrentProcess();
            _searchService = new MockSearchService();
            if (mathEngine is GpuMathEngine gpuEngine)
            {
                _syncGuard = gpuEngine.GetSyncGuard();
            }
        }

        public GenerativeNeuralNetworkLSTM? TrainModel(
            GenerativeNeuralNetworkLSTM initialModel,
            string datasetPath,
            string finalModelPath,
            float learningRate,
            int epochs,
            int batchSize,
            int contextWindowSize,
            float validationSplit)
        {
            int failedBatches = 0;
            if (!File.Exists(datasetPath))
                throw new FileNotFoundException("Arquivo de dataset não encontrado.", datasetPath);

            using (var datasetService = new DatasetService(Path.Combine(Environment.CurrentDirectory, "Dayson", "batches.bts")))
            {
                datasetService.InitializeAndSplit(File.ReadAllText(datasetPath), contextWindowSize,
                    initialModel.vocabularyManager.Vocab, "<PAD>", batchSize, validationSplit);

                GenerativeNeuralNetworkLSTM? currentModel = initialModel;
                TimeSpan totalElapsedTime = TimeSpan.Zero;
                var trainBatchIndices = datasetService.GetTrainBatchOffsets();
                var validationBatchIndices = datasetService.GetValidationBatchOffsets();

                for (int epoch = 0; epoch < epochs; epoch++)
                {
                    _stopwatch.Restart();
                    Console.WriteLine($"\n{'═',60}\nÉPOCA {epoch + 1}/{epochs} >> Learning Rate: {learningRate} >> {DateTime.UtcNow}\n{'═',60}");
                    
                    double totalEpochLoss = 0;
                    int batchCount = 0;

                    Console.WriteLine($"[Trainer] Carregando pesos do modelo para a VRAM para a Época {epoch + 1}...");
                    // Os pesos são carregados em um objeto que implementa IDisposable.
                    // O 'using' garante que eles sejam liberados da VRAM no final da época.
                    using (var weights = new ModelWeights(currentModel, _mathEngine, currentModel.GetTensorManager()))
                    {
                        Console.WriteLine("[Trainer] Pesos do modelo carregados na VRAM e prontos para a época.");

                        foreach (var batchIndex in trainBatchIndices)
                        {
                            List<(int[] InputIndices, int[] TargetIndices)>? batch = null;
                            try
                            {
                                batch = datasetService.LoadBatchFromDisk(batchIndex);
                                if (batch == null || !batch.Any()) continue;

                                double currentBatchLoss = 0;
                                foreach (var (inputIndices, targetIndices) in batch)
                                {
                                    if (inputIndices == null || !inputIndices.Any() || targetIndices == null || !targetIndices.Any()) continue;

                                    // ✅ CHAMADA FINAL E CORRETA:
                                    // TrainSequence agora gerencia seu próprio ciclo de vida de memória interna.
                                    currentBatchLoss += currentModel.TrainSequence(inputIndices, targetIndices, learningRate, weights);
                                }

                                totalEpochLoss += currentBatchLoss;
                                batchCount++;
                                double avgBatchLoss = currentBatchLoss / batch.Count;
                                Console.WriteLine($"Época: {epoch + 1}/{epochs} | Lotes: {batchCount}/{trainBatchIndices.Count} | Perda do Lote: {avgBatchLoss:F4}");

                                if (batchCount % 10 == 0) { MemoryWatchdog(); }
                            }
                            catch (Exception ex)
                            {
                                failedBatches++;
                                Console.ForegroundColor = ConsoleColor.Red;
                                Console.WriteLine($"[ERRO] Falha crítica no lote {batchIndex}: {ex.Message}\n{ex.StackTrace}");
                                Console.ResetColor();
                                if (failedBatches > 5) throw new Exception("Muitos lotes corrompidos. Abortando época.");
                            }
                            finally
                            {
                                if (batch != null) { CleanupSingleBatch(batch); }
                            }
                        }
                    } // 'using (weights)' garante que todos os tensores de peso sejam liberados da VRAM aqui.
                    
                    _stopwatch.Stop();
                    totalElapsedTime += _stopwatch.Elapsed;
                    double avgLoss = batchCount > 0 ? totalEpochLoss / (batchCount * batchSize) : double.PositiveInfinity;
                    string elapsedFormatted = $"{(int)_stopwatch.Elapsed.TotalHours:D2}:{_stopwatch.Elapsed.Minutes:D2}:{_stopwatch.Elapsed.Seconds:D2}";
                    Console.WriteLine($"\nÉpoca {epoch + 1}/{epochs} concluída. Perda média: {avgLoss:F4} | Duração: {elapsedFormatted}");
                    File.AppendAllText(logPath, $"Época {epoch + 1}: Perda Média: {avgLoss:F4}, Duração: {elapsedFormatted}\n");
                    
                    double validationLoss = ValidateModel(currentModel, datasetService, validationBatchIndices);
                    Console.WriteLine($"[Época {epoch + 1}] Perda Média de Validação: {validationLoss:F4}");
                    File.AppendAllText(logPath, $"[Época {epoch + 1}] Perda de Validação: {validationLoss:F4}\n");
                    
                    string modelPathForEpoch = Path.Combine(Path.GetDirectoryName(finalModelPath)!, $"dayson_{epoch + 1}.json");
                    FullMemoryReleaseCycle(ref currentModel, modelPathForEpoch, epochs, epoch, totalElapsedTime, finalModelPath);
                }
                
                if (currentModel == null && epochs > 0)
                {
                    Console.WriteLine("[Trainer] Recarregando modelo final para retorno...");
                    var vocabManager = new VocabularyManager();
                    vocabManager.LoadVocabulary();
                    string lastModelPath = Path.Combine(Path.GetDirectoryName(finalModelPath)!, $"dayson_{epochs}.json");
                    currentModel = GenerativeNeuralNetworkLSTM.Load(lastModelPath, _mathEngine, vocabManager, _searchService);
                }

                return currentModel;
            }
        }

        private void MemoryWatchdog()
        {
            long currentMemoryMB = GetCurrentMemoryUsageMB();
            if (currentMemoryMB > MEMORY_TRIM_THRESHOLD_MB && currentMemoryMB > (_lastTrimMemory + 512))
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine($"\n[Memory Watchdog] Uso de RAM: {currentMemoryMB}MB. Acionando limpeza preventiva...");
                Console.ResetColor();
                _syncGuard?.SynchronizeBeforeRead("MemoryWatchdog");
                ForceAggressiveGarbageCollection();
                _lastTrimMemory = GetCurrentMemoryUsageMB();
                Console.WriteLine($"[Memory Watchdog] Memória após limpeza: {_lastTrimMemory}MB");
            }
        }

        private void FullMemoryReleaseCycle(ref GenerativeNeuralNetworkLSTM? currentModel, string modelPathForEpoch, int epochs, int epoch, TimeSpan totalElapsedTime, string finalModelPath)
        {
            Console.WriteLine($"\n╔════ LIBERAÇÃO COMPLETA DE MEMÓRIA (Fim da Época {epoch + 1}) ════╗");
            long memoryBefore = GetCurrentMemoryUsageMB();
            if (currentModel != null)
            {
                currentModel.SaveModel(modelPathForEpoch);
                currentModel.ResetOptimizerState();
                currentModel.Dispose();
                currentModel = null;
            }
            ForceAggressiveGarbageCollection();
            long memoryAfter = GetCurrentMemoryUsageMB();
            Console.ForegroundColor = memoryBefore - memoryAfter > 0 ? ConsoleColor.Green : ConsoleColor.Yellow;
            Console.WriteLine($"[Resultado] Memória ANTES: {memoryBefore}MB → DEPOIS: {memoryAfter}MB | Liberada: {memoryBefore - memoryAfter}MB");
            Console.ResetColor();

            if (epoch < epochs - 1)
            {
                Console.WriteLine($"\nRecarregando modelo para Época {epoch + 2}...");
                var vocabManager = new VocabularyManager();
                vocabManager.LoadVocabulary();
                currentModel = GenerativeNeuralNetworkLSTM.Load(modelPathForEpoch, _mathEngine, vocabManager, _searchService);
                if (currentModel == null) throw new InvalidOperationException($"CRÍTICO: Falha ao recarregar o modelo {modelPathForEpoch}.");
                
                var avgEpochTime = TimeSpan.FromMilliseconds(totalElapsedTime.TotalMilliseconds / (epoch + 1));
                var estimatedTimeRemaining = TimeSpan.FromMilliseconds(avgEpochTime.TotalMilliseconds * (epochs - epoch - 1));
                Console.WriteLine($"[Estimativa] Tempo restante: ~{estimatedTimeRemaining:hh\\:mm\\:ss}");
            }
            Console.WriteLine("╚══════════════════════════════════════════════════════════╝\n");
        }

        private void CleanupSingleBatch(List<(int[] InputIndices, int[] TargetIndices)> batch)
        {
            batch.Clear();
            batch.TrimExcess();
        }

        private void ForceAggressiveGarbageCollection()
        {
            GC.Collect(2, GCCollectionMode.Forced, true, true);
            GC.WaitForPendingFinalizers();
        }

        private long GetCurrentMemoryUsageMB()
        {
            _currentProcess.Refresh();
            long currentMemory = _currentProcess.WorkingSet64 / (1024 * 1024);
            if (currentMemory > _peakMemoryUsageMB) _peakMemoryUsageMB = currentMemory;
            return currentMemory;
        }

        private double ValidateModel(GenerativeNeuralNetworkLSTM modelToValidate, DatasetService datasetService, List<long> validationBatchIndices)
        {
            Console.WriteLine("\n[Validação] Iniciando...");
            double totalLoss = 0;
            int sequenceCount = 0;
            var validationStopwatch = Stopwatch.StartNew();

            using (var weights = new ModelWeights(modelToValidate, _mathEngine, modelToValidate.GetTensorManager()))
            {
                foreach (var batchIndex in validationBatchIndices)
                {
                    var batch = datasetService.LoadBatchFromDisk(batchIndex);
                    if (batch == null) continue;
                    try
                    {
                        foreach (var (inputIndices, targetIndices) in batch)
                        {
                            totalLoss += modelToValidate.CalculateSequenceLoss(inputIndices, targetIndices);
                            sequenceCount++;
                        }
                    }
                    finally { CleanupSingleBatch(batch); }
                }
            }
            
            validationStopwatch.Stop();
            Console.WriteLine($"\r[Validação] Concluída em {validationStopwatch.Elapsed:mm\\:ss}. RAM: {GetCurrentMemoryUsageMB()}MB");
            return sequenceCount > 0 ? totalLoss / sequenceCount : double.PositiveInfinity;
        }
    }
}
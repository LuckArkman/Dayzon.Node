using System.Diagnostics;
using System.Text.Json;
using Galileu.Node.Core;
using Galileu.Node.Gpu;
using Galileu.Node.Interfaces;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;

namespace Galileu.Node.Brain
{
    public class NeuralNetworkLSTM : IDisposable
    {
        protected readonly AdamOptimizer _adamOptimizer;
        protected readonly IndividualFileTensorManager _tensorManager;
        protected readonly IMathEngine _mathEngine;
        public readonly DiskSwapManager _swapManager;
        protected readonly GpuSyncGuard? _syncGuard;

        private readonly int inputSize;
        private readonly int hiddenSize;
        public readonly int outputSize;
        private readonly string _sessionId;
        private bool _disposed = false;

        // ... (Declarações de IDs de peso) ...
        protected string _weightsEmbeddingId = null!;
        protected string _weightsInputForgetId = null!;
        protected string _weightsHiddenForgetId = null!;
        protected string _weightsInputInputId = null!;
        protected string _weightsHiddenInputId = null!;
        protected string _weightsInputCellId = null!;
        protected string _weightsHiddenCellId = null!;
        protected string _weightsInputOutputId = null!;
        protected string _weightsHiddenOutputId = null!;
        protected string _biasForgetId = null!;
        protected string _biasInputId = null!;
        protected string _biasCellId = null!;
        protected string _biasOutputId = null!;
        protected string _weightsHiddenOutputFinalId = null!;
        protected string _biasOutputFinalId = null!;
        protected string _hiddenStateId = null!;
        protected string _cellStateId = null!;
        protected string _lnForgetGammaId = null!;
        protected string _lnForgetBetaId = null!;
        protected string _lnInputGammaId = null!;
        protected string _lnInputBetaId = null!;
        protected string _lnCellGammaId = null!;
        protected string _lnCellBetaId = null!;
        protected string _lnOutputGammaId = null!;
        protected string _lnOutputBetaId = null!;
        protected string _forgetGateId = null!;
        protected string _inputGateId = null!;
        protected string _cellCandidateId = null!;
        protected string _outputGateId = null!;
        public int OutputSize => outputSize;


        public NeuralNetworkLSTM(int vocabSize, int embeddingSize, int hiddenSize, int outputSize, IMathEngine mathEngine)
        {
            this.inputSize = vocabSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;
            this._mathEngine = mathEngine ?? throw new ArgumentNullException(nameof(mathEngine));
            this._sessionId = $"session_{DateTime.UtcNow:yyyyMMdd_HHmmss}_{Guid.NewGuid():N}";

            this._swapManager = new DiskSwapManager(mathEngine, _sessionId);
            this._tensorManager = new IndividualFileTensorManager(mathEngine, _sessionId);
            this._adamOptimizer = new AdamOptimizer(_tensorManager);
            if (mathEngine is GpuMathEngine gpuEngine)
            {
                _syncGuard = gpuEngine.GetSyncGuard();
            }
            
            Console.WriteLine("╔═══════════════════════════════════════════════════════════╗");
            Console.WriteLine("║   ⚡ Memória Gerenciada com Offloading de Cache          ║");
            Console.WriteLine("╚═══════════════════════════════════════════════════════════╝");
            
            var rand = new Random(42);
            _weightsEmbeddingId = InitializeWeight(vocabSize, embeddingSize, rand, "WeightsEmbedding");
            _weightsInputForgetId = InitializeWeight(embeddingSize, hiddenSize, rand, "WeightsInputForget");
            _weightsHiddenForgetId = InitializeWeight(hiddenSize, hiddenSize, rand, "WeightsHiddenForget");
            _weightsInputInputId = InitializeWeight(embeddingSize, hiddenSize, rand, "WeightsInputInput");
            _weightsHiddenInputId = InitializeWeight(hiddenSize, hiddenSize, rand, "WeightsHiddenInput");
            _weightsInputCellId = InitializeWeight(embeddingSize, hiddenSize, rand, "WeightsInputCell");
            _weightsHiddenCellId = InitializeWeight(hiddenSize, hiddenSize, rand, "WeightsHiddenCell");
            _weightsInputOutputId = InitializeWeight(embeddingSize, hiddenSize, rand, "WeightsInputOutput");
            _weightsHiddenOutputId = InitializeWeight(hiddenSize, hiddenSize, rand, "WeightsHiddenOutput");
            _biasForgetId = InitializeWeight(1, hiddenSize, rand, "BiasForget");
            _biasInputId = InitializeWeight(1, hiddenSize, rand, "BiasInput");
            _biasCellId = InitializeWeight(1, hiddenSize, rand, "BiasCell");
            _biasOutputId = InitializeWeight(1, hiddenSize, rand, "BiasOutput");
            _weightsHiddenOutputFinalId = InitializeWeight(hiddenSize, outputSize, rand, "WeightsOutputFinal");
            _biasOutputFinalId = InitializeWeight(1, outputSize, rand, "BiasOutputFinal");
            _hiddenStateId = _tensorManager.CreateAndStoreZeros(new[] { 1, hiddenSize }, "HiddenState");
            _cellStateId = _tensorManager.CreateAndStoreZeros(new[] { 1, hiddenSize }, "CellState");
            _lnForgetGammaId = _tensorManager.CreateAndStore(Enumerable.Repeat(1.0f, hiddenSize).ToArray(), new[] { 1, hiddenSize }, "LN_Forget_Gamma");
            _lnForgetBetaId = _tensorManager.CreateAndStoreZeros(new[] { 1, hiddenSize }, "LN_Forget_Beta");
            _lnInputGammaId = _tensorManager.CreateAndStore(Enumerable.Repeat(1.0f, hiddenSize).ToArray(), new[] { 1, hiddenSize }, "LN_Input_Gamma");
            _lnInputBetaId = _tensorManager.CreateAndStoreZeros(new[] { 1, hiddenSize }, "LN_Input_Beta");
            _lnCellGammaId = _tensorManager.CreateAndStore(Enumerable.Repeat(1.0f, hiddenSize).ToArray(), new[] { 1, hiddenSize }, "LN_Cell_Gamma");
            _lnCellBetaId = _tensorManager.CreateAndStoreZeros(new[] { 1, hiddenSize }, "LN_Cell_Beta");
            _lnOutputGammaId = _tensorManager.CreateAndStore(Enumerable.Repeat(1.0f, hiddenSize).ToArray(), new[] { 1, hiddenSize }, "LN_Output_Gamma");
            _lnOutputBetaId = _tensorManager.CreateAndStoreZeros(new[] { 1, hiddenSize }, "LN_Output_Beta");
        }
        public GpuSyncGuard? GetSyncGuard()
        {
            return _syncGuard;
        }
        
        // ... (Construtor de Cópia) ...
        protected NeuralNetworkLSTM(NeuralNetworkLSTM existingModel) 
        {
            this.inputSize = existingModel.inputSize;
            this.hiddenSize = existingModel.hiddenSize;
            this.outputSize = existingModel.outputSize;
            this._mathEngine = existingModel._mathEngine;
            this._adamOptimizer = existingModel._adamOptimizer;
            this._sessionId = existingModel._sessionId;
            this._tensorManager = existingModel._tensorManager;
            this._swapManager = existingModel._swapManager;
            _weightsEmbeddingId = existingModel._weightsEmbeddingId;
            _weightsInputForgetId = existingModel._weightsInputForgetId;
            _weightsHiddenForgetId = existingModel._weightsHiddenForgetId;
            _weightsInputInputId = existingModel._weightsInputInputId;
            _weightsHiddenInputId = existingModel._weightsHiddenInputId;
            _weightsInputCellId = existingModel._weightsInputCellId;
            _weightsHiddenCellId = existingModel._weightsHiddenCellId;
            _weightsInputOutputId = existingModel._weightsInputOutputId;
            _weightsHiddenOutputId = existingModel._weightsHiddenOutputId;
            _biasForgetId = existingModel._biasForgetId;
            _biasInputId = existingModel._biasInputId;
            _biasCellId = existingModel._biasCellId;
            _biasOutputId = existingModel._biasOutputId;
            _weightsHiddenOutputFinalId = existingModel._weightsHiddenOutputFinalId;
            _biasOutputFinalId = existingModel._biasOutputFinalId;
            _hiddenStateId = existingModel._hiddenStateId;
            _cellStateId = existingModel._cellStateId;
            _lnForgetGammaId = existingModel._lnForgetGammaId;
            _lnForgetBetaId = existingModel._lnForgetBetaId;
            _lnInputGammaId = existingModel._lnInputGammaId;
            _lnInputBetaId = existingModel._lnInputBetaId;
            _lnCellGammaId = existingModel._lnCellGammaId;
            _lnCellBetaId = existingModel._lnCellBetaId;
            _lnOutputGammaId = existingModel._lnOutputGammaId;
            _lnOutputBetaId = existingModel._lnOutputBetaId;
        }

        private float[] CreateOrthogonalMatrix(int rows, int cols, Random rand)
        {
            var M = Matrix<float>.Build.Dense(rows, cols);
            var normalDist = new Normal(0, 1, rand);
            for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) M[i, j] = (float)normalDist.Sample();
            var svd = M.Svd(true);
            Matrix<float> orthogonalMatrix = rows >= cols ? svd.U : svd.VT.Transpose();
            if (orthogonalMatrix.RowCount != rows || orthogonalMatrix.ColumnCount != cols)
            {
                var finalMatrix = Matrix<float>.Build.Dense(rows, cols);
                finalMatrix.SetSubMatrix(0, 0, orthogonalMatrix.SubMatrix(0, Math.Min(rows, orthogonalMatrix.RowCount), 0, Math.Min(cols, orthogonalMatrix.ColumnCount)));
                return finalMatrix.ToColumnMajorArray();
            }
            return orthogonalMatrix.ToColumnMajorArray();
        }

        private string InitializeWeight(int rows, int cols, Random rand, string name)
        {
            float[] data;
            const float INITIALIZATION_VALUE_LIMIT = 0.5f;
            if (name.Contains("BiasForget"))
            {
                data = new float[rows * cols];
                Array.Fill(data, 1.0f);
                return _tensorManager.CreateAndStore(data, new[] { rows, cols }, name);
            }
            for (int attempt = 0; attempt < 100; attempt++)
            {
                if (name.Contains("WeightsHidden") && rows == cols) { data = CreateOrthogonalMatrix(rows, cols, rand); }
                else
                {
                    data = new float[rows * cols];
                    double limit = Math.Sqrt(6.0 / (rows + cols));
                    for (int i = 0; i < data.Length; i++) data[i] = (float)((rand.NextDouble() * 2 - 1) * limit);
                }
                if (!data.Any(v => float.IsNaN(v) || float.IsInfinity(v) || Math.Abs(v) > INITIALIZATION_VALUE_LIMIT))
                {
                    return _tensorManager.CreateAndStore(data, new[] { rows, cols }, name);
                }
            }
            throw new InvalidOperationException($"CRÍTICO: Falha ao inicializar '{name}'.");
        }
        
        protected class GateCache
        {
            public string Term1Id { get; set; } = string.Empty;
            public string Term2Id { get; set; } = string.Empty;
        }
        
        protected class StepForwardCache
        {
            public GateCache ForgetGateCache { get; set; } = new GateCache();
            public GateCache InputGateCache { get; set; } = new GateCache();
            public GateCache CellCandidateCache { get; set; } = new GateCache();
            public GateCache OutputGateCache { get; set; } = new GateCache();
            public string InputId { get; set; } = string.Empty;
            public string HiddenPrevId { get; set; } = string.Empty;
            public string CellPrevId { get; set; } = string.Empty;
            public string CellNextId { get; set; } = string.Empty;
            public string HiddenNextId { get; set; } = string.Empty;
            public string TanhCellNextId { get; set; } = string.Empty;
        }

        private GateCache ComputeAndCacheTerms(IMathTensor input, IMathTensor h_prev, IMathTensor W_i, IMathTensor W_h, TensorScope scope, string gateName, int t)
        {
            var cache = new GateCache();
            using (var term1 = scope.CreateTensor(new[] { 1, hiddenSize }))
            using (var term2 = scope.CreateTensor(new[] { 1, hiddenSize }))
            {
                _mathEngine.MatrixMultiply(input, W_i, term1);
                _mathEngine.MatrixMultiply(h_prev, W_h, term2);
                cache.Term1Id = _swapManager.SwapOut(term1, $"term1_{gateName}_t{t}");
                cache.Term2Id = _swapManager.SwapOut(term2, $"term2_{gateName}_t{t}");
            }
            return cache;
        }

        private void ComputeGateWithCache(GateCache cache, IMathTensor bias, IMathTensor ln_gamma, IMathTensor ln_beta, IMathTensor result, TensorScope scope)
        {
            using (var term1 = scope.Track(_swapManager.LoadFromSwap(cache.Term1Id)))
            using (var term2 = scope.Track(_swapManager.LoadFromSwap(cache.Term2Id)))
            using (var linear = scope.CreateTensor(result.Shape))
            {
                _mathEngine.Add(term1, term2, linear);
                _mathEngine.AddBroadcast(linear, bias, linear);
                _mathEngine.LayerNorm(linear, ln_gamma, ln_beta);
                _mathEngine.Sigmoid(linear, result);
            }
        }

        private void ComputeCellCandidateWithCache(GateCache cache, IMathTensor bias, IMathTensor ln_gamma, IMathTensor ln_beta, IMathTensor result, TensorScope scope)
        {
            using (var term1 = scope.Track(_swapManager.LoadFromSwap(cache.Term1Id)))
            using (var term2 = scope.Track(_swapManager.LoadFromSwap(cache.Term2Id)))
            using (var linear = scope.CreateTensor(result.Shape))
            {
                _mathEngine.Add(term1, term2, linear);
                _mathEngine.AddBroadcast(linear, bias, linear);
                _mathEngine.LayerNorm(linear, ln_gamma, ln_beta);
                _mathEngine.Tanh(linear, result);
            }
        }
        
        /// <summary>
        /// ✅ VERSÃO CORRIGIDA - TrainSequence com gerenciamento completo de memória
        /// 
        /// CORREÇÕES IMPLEMENTADAS:
        /// 1. ForwardCache descartado explicitamente
        /// 2. Gradientes na RAM/VRAM liberados após Adam
        /// 3. Sincronização GPU antes de deletar arquivos
        /// 4. Ordem de limpeza corrigida: RAM → VRAM → Disco
        /// 5. Rastreamento de todos os tensores temporários
        /// </summary>
        public float TrainSequence(int[] inputIndices, int[] targetIndices, 
            float learningRate, ModelWeights weights)
        {
            var allTempFiles = new List<string>();
            var allTempTensors = new List<IMathTensor>(); // ✅ NOVO: Rastreia tensores RAM
            float loss = 0f;
            
            try
            {
                // ═══════════════════════════════════════════════════════════
                // FASE 1: FORWARD PASS
                // ═══════════════════════════════════════════════════════════
                
                var (computedLoss, forwardCache, forwardFiles) = 
                    ForwardPassWithOffloading(inputIndices, targetIndices, weights);
                
                loss = computedLoss;
                allTempFiles.AddRange(forwardFiles);
                
                // ═══════════════════════════════════════════════════════════
                // FASE 2: BACKWARD PASS
                // ═══════════════════════════════════════════════════════════
                
                var (gradIds, gradFiles) = 
                    BackwardPassWithOffloading(inputIndices, targetIndices, forwardCache, weights);
                
                allTempFiles.AddRange(gradFiles);
                
                // ✅ NOVO: Libera forwardCache IMEDIATAMENTE após backward
                // (não precisa mais dele)
                CleanupForwardCache(forwardCache);
                
                // ═══════════════════════════════════════════════════════════
                // FASE 3: ADAM UPDATE
                // ═══════════════════════════════════════════════════════════
                
                var weightIds = new Dictionary<string, string>
                {
                    // Embedding
                    { "W_embed", _weightsEmbeddingId },
                    
                    // Forget gate
                    { "W_if", _weightsInputForgetId },
                    { "W_hf", _weightsHiddenForgetId },
                    { "B_f", _biasForgetId },
                    
                    // Input gate
                    { "W_ii", _weightsInputInputId },
                    { "W_hi", _weightsHiddenInputId },
                    { "B_i", _biasInputId },
                    
                    // Cell candidate
                    { "W_ic", _weightsInputCellId },
                    { "W_hc", _weightsHiddenCellId },
                    { "B_c", _biasCellId },
                    
                    // Output gate
                    { "W_io", _weightsInputOutputId },
                    { "W_ho", _weightsHiddenOutputId },
                    { "B_o", _biasOutputId },
                    
                    // Output layer
                    { "W_hy", _weightsHiddenOutputFinalId },
                    { "B_y", _biasOutputFinalId },
                    
                    // LayerNorm parameters
                    { "LN_f_gamma", _lnForgetGammaId },
                    { "LN_f_beta", _lnForgetBetaId },
                    { "LN_i_gamma", _lnInputGammaId },
                    { "LN_i_beta", _lnInputBetaId },
                    { "LN_c_gamma", _lnCellGammaId },
                    { "LN_c_beta", _lnCellBetaId },
                    { "LN_o_gamma", _lnOutputGammaId },
                    { "LN_o_beta", _lnOutputBetaId }
                };
                
                // ✅ MODIFICADO: UpdateAdam agora rastreia tensores temporários
                var adamTempTensors = UpdateAdamGPUPassZeroRAMWithTracking(weightIds, gradIds);
                allTempTensors.AddRange(adamTempTensors);
                
                // ✅ NOVO: Libera gradientes IMEDIATAMENTE após Adam
                // (já foram aplicados nos pesos)
                CleanupGradients(gradIds);
                
                return loss;
            }
            finally
            {
                // ═══════════════════════════════════════════════════════════
                // LIMPEZA COMPLETA - ORDEM CRÍTICA
                // ═══════════════════════════════════════════════════════════
                
                try
                {
                    // ✅ PASSO 1: Sincroniza GPU ANTES de qualquer limpeza
                    if (_mathEngine.IsGpu && _syncGuard != null)
                    {
                        _syncGuard.SynchronizeBeforeRead("TrainSequenceCleanup");
                    }
                    
                    // ✅ PASSO 2: Libera tensores da RAM/VRAM primeiro
                    foreach (var tensor in allTempTensors)
                    {
                        try
                        {
                            tensor?.Dispose();
                        }
                        catch (Exception ex)
                        {
                            #if DEBUG
                            Console.WriteLine($"[TrainSequence] Erro ao dispor tensor: {ex.Message}");
                            #endif
                        }
                    }
                    allTempTensors.Clear();
                    
                    // ✅ PASSO 3: Remove arquivos de swap do disco
                    foreach (var fileId in allTempFiles.Distinct()) // .Distinct() evita duplas
                    {
                        try
                        {
                            _swapManager.DeleteSwapFile(fileId);
                        }
                        catch (Exception ex)
                        {
                            #if DEBUG
                            Console.WriteLine($"[TrainSequence] Erro ao deletar swap: {ex.Message}");
                            #endif
                        }
                    }
                    
                    // ✅ PASSO 4: Remove metadados dos tensores
                    foreach (var fileId in allTempFiles.Distinct())
                    {
                        try
                        {
                            _tensorManager.DeleteTensor(fileId);
                        }
                        catch (Exception ex)
                        {
                            #if DEBUG
                            Console.WriteLine($"[TrainSequence] Erro ao deletar tensor metadata: {ex.Message}");
                            #endif
                        }
                    }
                    
                    allTempFiles.Clear();
                }
                catch (Exception ex)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"[TrainSequence] ERRO CRÍTICO na limpeza: {ex.Message}");
                    Console.ResetColor();
                    
                    // ✅ IMPORTANTE: Não re-lança exceção aqui
                    // Melhor vazar memória do que crashar o treino
                }
            }
        }

        /// <summary>
        /// ✅ IMPLEMENTAÇÃO COMPLETA - Limpa cache do forward pass
        /// 
        /// Esta versão limpa TODOS os tensores do cache de forma segura,
        /// liberando tanto memória RAM quanto VRAM.
        /// 
        /// IMPORTANTE: StepForwardCache armazena IDs de arquivos em disco.
        /// Os arquivos serão deletados no finally do TrainSequence,
        /// mas precisamos liberar as estruturas em memória aqui.
        /// </summary>
        private void CleanupForwardCache(List<StepForwardCache> forwardCache)
        {
            if (forwardCache == null || forwardCache.Count == 0)
                return;
    
            try
            {
                foreach (var cache in forwardCache)
                {
                    if (cache == null) continue;
            
                    // Limpa IDs (strings)
                    cache.InputId = string.Empty;
                    cache.HiddenPrevId = string.Empty;
                    cache.CellPrevId = string.Empty;
                    cache.CellNextId = string.Empty;
                    cache.HiddenNextId = string.Empty;
                    cache.TanhCellNextId = string.Empty;
            
                    // Recria GateCaches vazios (força garbage collection dos antigos)
                    cache.ForgetGateCache = new GateCache();
                    cache.InputGateCache = new GateCache();
                    cache.CellCandidateCache = new GateCache();
                    cache.OutputGateCache = new GateCache();
                }
        
                // Limpa a lista
                forwardCache.Clear();
        
#if DEBUG
                Console.WriteLine($"[CleanupForwardCache] ✓ Cache limpo");
#endif
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[CleanupForwardCache] ⚠️ Erro: {ex.Message}");
                try { forwardCache?.Clear(); } catch { }
            }
        }

        /// <summary>
        /// ✅ NOVO: Limpa dicionário de gradientes
        /// </summary>
        private void CleanupGradients(Dictionary<string, string> gradIds)
        {
            if (gradIds == null) return;
            
            try
            {
                // Se há tensores carregados em RAM (não deveria, mas por segurança)
                // Os IDs apontam para arquivos em disco que serão deletados no finally
                gradIds.Clear();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Cleanup] Erro ao limpar gradientes: {ex.Message}");
            }
        }

        /// <summary>
        /// ✅ NOVO: Versão do Adam que rastreia tensores temporários
        /// </summary>
        private List<IMathTensor> UpdateAdamGPUPassZeroRAMWithTracking(
            Dictionary<string, string> weightIds, 
            Dictionary<string, string> gradIds)
        {
            var tempTensors = new List<IMathTensor>();
            
            foreach (var (paramName, weightId) in weightIds)
            {
                if (!gradIds.TryGetValue(paramName, out string gradId))
                {
                    continue; // Sem gradiente para este parâmetro
                }
                
                // Adam update acontece internamente no AdamOptimizer
                // que já gerencia seus próprios tensores m/v em disco
                
                // ✅ IMPORTANTE: Obter ID da layer para Adam
                int layerId = GetLayerIdFromParamName(paramName);
                
                try
                {
                    // Carrega parâmetros do disco para VRAM (temporariamente)
                    using (var scope = new TensorScope($"Adam_{paramName}", _mathEngine, _tensorManager))
                    {
                        var parameters = scope.LoadTensor(weightId);
                        var gradients = scope.LoadTensor(gradId);
                        
                        // Adam gerencia m/v internamente (já está em disco)
                        _adamOptimizer.UpdateParametersGpu(layerId, parameters, gradients, _mathEngine);
                        
                        // Salva parâmetros atualizados de volta no disco
                        _tensorManager.OverwriteTensor(weightId, parameters);
                        
                        // ✅ TensorScope.Dispose() garante limpeza de parameters e gradients
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[Adam] Erro ao atualizar {paramName}: {ex.Message}");
                }
            }
            
            return tempTensors; // Vazio porque TensorScope gerencia tudo
        }

        /// <summary>
        /// ✅ NOVO: Mapeia nome do parâmetro para ID de layer (para Adam)
        /// </summary>
        private int GetLayerIdFromParamName(string paramName)
        {
            // Mapeamento simples baseado no nome do parâmetro
            return paramName switch
            {
                "W_embed" => 0,
                "W_if" or "W_hf" or "B_f" or "LN_f_gamma" or "LN_f_beta" => 1, // Forget gate
                "W_ii" or "W_hi" or "B_i" or "LN_i_gamma" or "LN_i_beta" => 2, // Input gate
                "W_ic" or "W_hc" or "B_c" or "LN_c_gamma" or "LN_c_beta" => 3, // Cell
                "W_io" or "W_ho" or "B_o" or "LN_o_gamma" or "LN_o_beta" => 4, // Output gate
                "W_hy" or "B_y" => 5, // Output layer
                _ => 99 // Fallback
            };
        }

        protected (float, List<StepForwardCache>, List<string>) ForwardPassWithOffloading(int[] inputIndices, int[] targetIndices, ModelWeights weights)
        {
            float sequenceLoss = 0;
            var forwardCache = new List<StepForwardCache>();
            var generatedFiles = new List<string>();
            using var sequenceScope = new TensorScope("ForwardPass", _mathEngine, _tensorManager);
            var h_prev = sequenceScope.Track(_tensorManager.LoadTensor(_hiddenStateId));
            var c_prev = sequenceScope.Track(_tensorManager.LoadTensor(_cellStateId));

            for (int t = 0; t < inputIndices.Length; t++)
            {
                using var stepScope = sequenceScope.CreateSubScope($"Forward_t{t}");
                var stepCache = new StepForwardCache();
                
                var input = stepScope.CreateTensor(new[] { 1, weights.Embedding.Shape[1] });
                _mathEngine.Lookup(weights.Embedding, inputIndices[t], input);
                stepCache.InputId = _swapManager.SwapOut(input, $"input_t{t}");
                stepCache.HiddenPrevId = _swapManager.SwapOut(h_prev, $"h_prev_t{t}");
                stepCache.CellPrevId = _swapManager.SwapOut(c_prev, $"c_prev_t{t}");

                stepCache.ForgetGateCache = ComputeAndCacheTerms(input, h_prev, weights.W_if, weights.W_hf, stepScope, "fg", t);
                stepCache.InputGateCache = ComputeAndCacheTerms(input, h_prev, weights.W_ii, weights.W_hi, stepScope, "ig", t);
                stepCache.CellCandidateCache = ComputeAndCacheTerms(input, h_prev, weights.W_ic, weights.W_hc, stepScope, "cc", t);
                stepCache.OutputGateCache = ComputeAndCacheTerms(input, h_prev, weights.W_io, weights.W_ho, stepScope, "og", t);

                var c_next = stepScope.CreateTensor(new[] { 1, hiddenSize });
                using (var fg = stepScope.CreateTensor(new[] { 1, hiddenSize }))
                using (var ig = stepScope.CreateTensor(new[] { 1, hiddenSize }))
                using (var cc = stepScope.CreateTensor(new[] { 1, hiddenSize }))
                using (var term1 = stepScope.CreateTensor(new[] { 1, hiddenSize }))
                using (var term2 = stepScope.CreateTensor(new[] { 1, hiddenSize }))
                {
                    ComputeGateWithCache(stepCache.ForgetGateCache, weights.B_f, weights.LN_f_gamma, weights.LN_f_beta, fg, stepScope);
                    ComputeGateWithCache(stepCache.InputGateCache, weights.B_i, weights.LN_i_gamma, weights.LN_i_beta, ig, stepScope);
                    ComputeCellCandidateWithCache(stepCache.CellCandidateCache, weights.B_c, weights.LN_c_gamma, weights.LN_c_beta, cc, stepScope);
                    _mathEngine.Multiply(fg, c_prev, term1);
                    _mathEngine.Multiply(ig, cc, term2);
                    _mathEngine.Add(term1, term2, c_next);
                }
                stepCache.CellNextId = _swapManager.SwapOut(c_next, $"c_next_t{t}");

                var h_next = stepScope.CreateTensor(new[] { 1, hiddenSize });
                using (var og = stepScope.CreateTensor(new[] { 1, hiddenSize }))
                using (var tanh_c = stepScope.CreateTensor(new[] { 1, hiddenSize }))
                {
                    ComputeGateWithCache(stepCache.OutputGateCache, weights.B_o, weights.LN_o_gamma, weights.LN_o_beta, og, stepScope);
                    _mathEngine.Tanh(c_next, tanh_c);
                    stepCache.TanhCellNextId = _swapManager.SwapOut(tanh_c, $"tanh_c_t{t}");
                    _mathEngine.Multiply(og, tanh_c, h_next);
                }
                stepCache.HiddenNextId = _swapManager.SwapOut(h_next, $"h_next_t{t}");

                using (var pred = stepScope.CreateTensor(new[] { 1, outputSize }))
                using (var outLinear = stepScope.CreateTensor(new[] { 1, outputSize }))
                {
                    _mathEngine.MatrixMultiply(h_next, weights.W_hy, outLinear);
                    _mathEngine.AddBroadcast(outLinear, weights.B_y, outLinear);
                    _mathEngine.Softmax(outLinear, pred);
                    using (var predCpu = pred.ToCpuTensor()) { sequenceLoss += -MathF.Log(Math.Max(predCpu.GetData()[targetIndices[t]], 1e-9f)); }
                }
                
                h_prev.Dispose();
                c_prev.Dispose();
                h_prev = sequenceScope.Track(_mathEngine.Clone(h_next));
                c_prev = sequenceScope.Track(_mathEngine.Clone(c_next));
                
                forwardCache.Add(stepCache);
                generatedFiles.AddRange(new[] { stepCache.InputId, stepCache.HiddenPrevId, stepCache.CellPrevId, stepCache.CellNextId, stepCache.HiddenNextId, stepCache.TanhCellNextId, stepCache.ForgetGateCache.Term1Id, stepCache.ForgetGateCache.Term2Id, stepCache.InputGateCache.Term1Id, stepCache.InputGateCache.Term2Id, stepCache.CellCandidateCache.Term1Id, stepCache.CellCandidateCache.Term2Id, stepCache.OutputGateCache.Term1Id, stepCache.OutputGateCache.Term2Id });
            }
            _tensorManager.OverwriteTensor(_hiddenStateId, h_prev);
            _tensorManager.OverwriteTensor(_cellStateId, c_prev);
            return (sequenceLoss / inputIndices.Length, forwardCache, generatedFiles);
        }
        
        protected (Dictionary<string, string> gradIds, List<string> gradFiles) BackwardPassWithOffloading(
    int[] inputIndices, int[] targetIndices, List<StepForwardCache> forwardCache, ModelWeights weights)
{
    var gradFiles = new List<string>();
    var gradIds = new Dictionary<string, string>();

    using(var sequenceScope = new TensorScope("BackwardSequence", _mathEngine, _tensorManager))
    {
        var inMemoryGrads = new Dictionary<string, IMathTensor>
        {
            ["W_embedding"] = sequenceScope.CreateTensor(weights.Embedding.Shape), ["W_if"] = sequenceScope.CreateTensor(weights.W_if.Shape), ["W_hf"] = sequenceScope.CreateTensor(weights.W_hf.Shape), ["B_f"] = sequenceScope.CreateTensor(weights.B_f.Shape),
            ["W_ii"] = sequenceScope.CreateTensor(weights.W_ii.Shape), ["W_hi"] = sequenceScope.CreateTensor(weights.W_hi.Shape), ["B_i"] = sequenceScope.CreateTensor(weights.B_i.Shape),
            ["W_ic"] = sequenceScope.CreateTensor(weights.W_ic.Shape), ["W_hc"] = sequenceScope.CreateTensor(weights.W_hc.Shape), ["B_c"] = sequenceScope.CreateTensor(weights.B_c.Shape),
            ["W_io"] = sequenceScope.CreateTensor(weights.W_io.Shape), ["W_ho"] = sequenceScope.CreateTensor(weights.W_ho.Shape), ["B_o"] = sequenceScope.CreateTensor(weights.B_o.Shape),
            ["W_hy"] = sequenceScope.CreateTensor(weights.W_hy.Shape), ["B_y"] = sequenceScope.CreateTensor(weights.B_y.Shape),
            ["LN_f_gamma"] = sequenceScope.CreateTensor(weights.LN_f_gamma.Shape), ["LN_f_beta"] = sequenceScope.CreateTensor(weights.LN_f_beta.Shape), ["LN_i_gamma"] = sequenceScope.CreateTensor(weights.LN_i_gamma.Shape), ["LN_i_beta"] = sequenceScope.CreateTensor(weights.LN_i_beta.Shape),
            ["LN_c_gamma"] = sequenceScope.CreateTensor(weights.LN_c_gamma.Shape), ["LN_c_beta"] = sequenceScope.CreateTensor(weights.LN_c_beta.Shape), ["LN_o_gamma"] = sequenceScope.CreateTensor(weights.LN_o_gamma.Shape), ["LN_o_beta"] = sequenceScope.CreateTensor(weights.LN_o_beta.Shape),
        };
        
        var dh_next = sequenceScope.CreateTensor(new[] { 1, hiddenSize });
        var dc_next = sequenceScope.CreateTensor(new[] { 1, hiddenSize });

        for (int t = targetIndices.Length - 1; t >= 0; t--)
        {
            using (var bpttScope = sequenceScope.CreateSubScope($"BPTT_t{t}"))
            {
                var cache = forwardCache[t];
                var input = bpttScope.Track(_swapManager.LoadFromSwap(cache.InputId));
                var h_prev = bpttScope.Track(_swapManager.LoadFromSwap(cache.HiddenPrevId));
                var c_prev = bpttScope.Track(_swapManager.LoadFromSwap(cache.CellPrevId));
                var c_next = bpttScope.Track(_swapManager.LoadFromSwap(cache.CellNextId));
                var h_next = bpttScope.Track(_swapManager.LoadFromSwap(cache.HiddenNextId));
                var tanh_c = bpttScope.Track(_swapManager.LoadFromSwap(cache.TanhCellNextId));
                
                using var fg = bpttScope.CreateTensor(new[] { 1, hiddenSize });
                ComputeGateWithCache(cache.ForgetGateCache, weights.B_f, weights.LN_f_gamma, weights.LN_f_beta, fg, bpttScope);
                using var ig = bpttScope.CreateTensor(new[] { 1, hiddenSize });
                ComputeGateWithCache(cache.InputGateCache, weights.B_i, weights.LN_i_gamma, weights.LN_i_beta, ig, bpttScope);
                using var cc = bpttScope.CreateTensor(new[] { 1, hiddenSize });
                ComputeCellCandidateWithCache(cache.CellCandidateCache, weights.B_c, weights.LN_c_gamma, weights.LN_c_beta, cc, bpttScope);
                using var og = bpttScope.CreateTensor(new[] { 1, hiddenSize });
                ComputeGateWithCache(cache.OutputGateCache, weights.B_o, weights.LN_o_gamma, weights.LN_o_beta, og, bpttScope);

                using var pred = bpttScope.CreateTensor(new[] { 1, outputSize });
                using (var outLinear = bpttScope.CreateTensor(new[] { 1, outputSize }))
                {
                    _mathEngine.MatrixMultiply(h_next, weights.W_hy, outLinear);
                    _mathEngine.AddBroadcast(outLinear, weights.B_y, outLinear);
                    _mathEngine.Softmax(outLinear, pred);
                }
                
                var d_pred = bpttScope.Track(_mathEngine.Clone(pred));
                using (var oneHot = _mathEngine.CreateOneHotTensor(new[] { targetIndices[t] }, outputSize)) { if (oneHot != null) _mathEngine.Subtract(d_pred, oneHot, d_pred); }
                
                using(var d_Why = bpttScope.CreateTensor(weights.W_hy.Shape))
                {
                     _mathEngine.MatrixMultiplyTransposeA(h_next, d_pred, d_Why);
                     _mathEngine.Add(inMemoryGrads["W_hy"], d_Why, inMemoryGrads["W_hy"]);
                }
                _mathEngine.Add(inMemoryGrads["B_y"], d_pred, inMemoryGrads["B_y"]);

                using var dh = bpttScope.CreateTensor(h_next.Shape);
                _mathEngine.MatrixMultiplyTransposeB(d_pred, weights.W_hy, dh);
                _mathEngine.Add(dh, dh_next, dh);

                using var d_tanh_c = bpttScope.CreateTensor(tanh_c.Shape);
                _mathEngine.Multiply(dh, og, d_tanh_c);
                _mathEngine.TanhDerivative(tanh_c, d_tanh_c);
                
                using var dc = bpttScope.CreateTensor(c_prev.Shape);
                _mathEngine.Add(dc_next, d_tanh_c, dc);

                // ✅ CORREÇÃO DE SINTAXE APLICADA AQUI
                using var d_og = bpttScope.CreateTensor(og.Shape);
                using var d_cc = bpttScope.CreateTensor(cc.Shape);
                using var d_ig = bpttScope.CreateTensor(ig.Shape);
                using var d_fg = bpttScope.CreateTensor(fg.Shape);
                
                _mathEngine.Multiply(dh, tanh_c, d_og);
                _mathEngine.SigmoidDerivative(og, d_og);
                _mathEngine.Multiply(dc, ig, d_cc);
                _mathEngine.TanhDerivative(cc, d_cc);
                _mathEngine.Multiply(dc, cc, d_ig);
                _mathEngine.SigmoidDerivative(ig, d_ig);
                _mathEngine.Multiply(dc, c_prev, d_fg);
                _mathEngine.SigmoidDerivative(fg, d_fg);

                _mathEngine.Multiply(dc, fg, dc_next);

                var d_h_prev = bpttScope.CreateTensor(h_prev.Shape);
                var d_input_acc = bpttScope.CreateTensor(input.Shape);

                Action<IMathTensor, IMathTensor, IMathTensor, string, string, string> backwardGate = (d_gate, W_h, W_i, grad_Wh_key, grad_Wi_key, grad_B_key) =>
                {
                    using var gateScope = bpttScope.CreateSubScope("BPTT_Gate");
                    
                    // ✅ CORREÇÃO DE SINTAXE APLICADA AQUI
                    using var d_Wh = gateScope.CreateTensor(W_h.Shape);
                    using var d_Wi = gateScope.CreateTensor(W_i.Shape);
                    
                    _mathEngine.MatrixMultiplyTransposeA(h_prev, d_gate, d_Wh);
                    _mathEngine.Add(inMemoryGrads[grad_Wh_key], d_Wh, inMemoryGrads[grad_Wh_key]);
                    _mathEngine.MatrixMultiplyTransposeA(input, d_gate, d_Wi);
                    _mathEngine.Add(inMemoryGrads[grad_Wi_key], d_Wi, inMemoryGrads[grad_Wi_key]);
                    
                    _mathEngine.Add(inMemoryGrads[grad_B_key], d_gate, inMemoryGrads[grad_B_key]);

                    // ✅ CORREÇÃO DE SINTAXE APLICADA AQUI
                    using var dh_prev_contrib = gateScope.CreateTensor(h_prev.Shape);
                    using var d_input_contrib = gateScope.CreateTensor(input.Shape);
                    
                    _mathEngine.MatrixMultiplyTransposeB(d_gate, W_h, dh_prev_contrib);
                    _mathEngine.Add(d_h_prev, dh_prev_contrib, d_h_prev);
                    _mathEngine.MatrixMultiplyTransposeB(d_gate, W_i, d_input_contrib);
                    _mathEngine.Add(d_input_acc, d_input_contrib, d_input_acc);
                };
                backwardGate(d_fg, weights.W_hf, weights.W_if, "W_hf", "W_if", "B_f");
                backwardGate(d_ig, weights.W_hi, weights.W_ii, "W_hi", "W_ii", "B_i");
                backwardGate(d_cc, weights.W_hc, weights.W_ic, "W_hc", "W_ic", "B_c");
                backwardGate(d_og, weights.W_ho, weights.W_io, "W_ho", "W_io", "B_o");

                _mathEngine.Add(dh_next, d_h_prev, dh_next);
                _mathEngine.AccumulateGradient(inMemoryGrads["W_embedding"], d_input_acc, inputIndices[t]);
            }
        }

        ApplyGlobalGradientClipping(inMemoryGrads.Values);
        
        foreach(var kvp in inMemoryGrads)
        {
            var gradId = _tensorManager.StoreTensor(kvp.Value, $"grad_{kvp.Key}");
            gradIds[kvp.Key] = gradId;
            gradFiles.Add(gradId);
        }
    }
    return (gradIds, gradFiles);
}

        protected void UpdateAdamGPUPassZeroRAM(Dictionary<string, string> weightIds, Dictionary<string, string> gradIds)
        {
            using (var updateScope = new TensorScope("AdamUpdate_FullPass", _mathEngine, _tensorManager))
            {
                int layerIndex = 0;
                foreach (var kvp in weightIds)
                {
                    if (!gradIds.TryGetValue(kvp.Key, out var gradId)) continue;
                    var paramTensor = updateScope.LoadTensor(kvp.Value);
                    var gradTensor = updateScope.LoadTensor(gradId);
                    _adamOptimizer.UpdateParametersGpu(layerIndex, paramTensor, gradTensor, _mathEngine);
                    _tensorManager.OverwriteTensor(kvp.Value, paramTensor);
                    layerIndex++;
                }
            }
        }
        
        private void ApplyGlobalGradientClipping(IEnumerable<IMathTensor> gradients, float clipValue = 5.0f, float maxNorm = 30.0f)
        {
            foreach (var grad in gradients) { _mathEngine.SanitizeAndClip(grad, clipValue); }
            double totalSumOfSquares = gradients.Sum(grad => _mathEngine.CalculateSumOfSquares(grad));
            float totalNorm = MathF.Sqrt((float)totalSumOfSquares);
            if (totalNorm <= maxNorm) return;
            float scaleFactor = maxNorm / (totalNorm + 1e-8f);
            foreach (var grad in gradients) { _mathEngine.Scale(grad, scaleFactor); }
        }
        
        public (float, List<string>) RunForwardPassForInference(int[] inputIndices, int[] targetIndices, ModelWeights weights)
        {
            var (loss, _, _) = ForwardPassWithOffloading(inputIndices, targetIndices, weights);
            return (loss, new List<string>());
        }
        
        public void ResetHiddenState()
        {
            var zeros = new float[hiddenSize];
            _tensorManager.UpdateTensor(_hiddenStateId, t => t.UpdateFromCpu(zeros));
            _tensorManager.UpdateTensor(_cellStateId, t => t.UpdateFromCpu(zeros));
        }

        public void SaveModel(string filePath)
        {
            var embeddingSize = _tensorManager.GetShape(_weightsEmbeddingId)[1];

            var modelData = new
            {
                VocabSize = inputSize,
                EmbeddingSize = embeddingSize,
                HiddenSize = hiddenSize,
                OutputSize = outputSize,
                SessionId = _sessionId,
                TensorIds = new Dictionary<string, string>
                {
                    ["WeightsEmbedding"] = _weightsEmbeddingId,
                    ["WeightsInputForget"] = _weightsInputForgetId,
                    ["WeightsHiddenForget"] = _weightsHiddenForgetId,
                    ["WeightsInputInput"] = _weightsInputInputId,
                    ["WeightsHiddenInput"] = _weightsHiddenInputId,
                    ["WeightsInputCell"] = _weightsInputCellId,
                    ["WeightsHiddenCell"] = _weightsHiddenCellId,
                    ["WeightsInputOutput"] = _weightsInputOutputId,
                    ["WeightsHiddenOutput"] = _weightsHiddenOutputId,
                    ["BiasForget"] = _biasForgetId,
                    ["BiasInput"] = _biasInputId,
                    ["BiasCell"] = _biasCellId,
                    ["BiasOutput"] = _biasOutputId,
                    ["WeightsOutputFinal"] = _weightsHiddenOutputFinalId,
                    ["BiasOutputFinal"] = _biasOutputFinalId,
                    ["HiddenState"] = _hiddenStateId,
                    ["CellState"] = _cellStateId,
                    // Garante que os pesos de Layer Normalization também sejam salvos
                    ["LN_Forget_Gamma"] = _lnForgetGammaId,
                    ["LN_Forget_Beta"] = _lnForgetBetaId,
                    ["LN_Input_Gamma"] = _lnInputGammaId,
                    ["LN_Input_Beta"] = _lnInputBetaId,
                    ["LN_Cell_Gamma"] = _lnCellGammaId,
                    ["LN_Cell_Beta"] = _lnCellBetaId,
                    ["LN_Output_Gamma"] = _lnOutputGammaId,
                    ["LN_Output_Beta"] = _lnOutputBetaId
                }
            };

            string jsonString = JsonSerializer.Serialize(modelData, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(filePath, jsonString);
        }

        public static NeuralNetworkLSTM? LoadModel(string filePath, IMathEngine mathEngine)
{
    if (!File.Exists(filePath))
    {
        Console.WriteLine($"[LoadModel] Erro: Arquivo do modelo não encontrado em '{filePath}'.");
        return null;
    }

    try
    {
        string jsonString = File.ReadAllText(filePath);
        using var doc = JsonDocument.Parse(jsonString);
        var root = doc.RootElement;

        int vocabSize = root.GetProperty("VocabSize").GetInt32();
        int embeddingSize = root.GetProperty("EmbeddingSize").GetInt32();
        int hiddenSize = root.GetProperty("HiddenSize").GetInt32();
        int outputSize = root.GetProperty("OutputSize").GetInt32();
        
        // Usa o construtor que inicializa a arquitetura, mas não os pesos.
        var model = new NeuralNetworkLSTM(vocabSize, embeddingSize, hiddenSize, outputSize, mathEngine);

        var tensorIds = root.GetProperty("TensorIds");
        // Atribui os IDs dos tensores salvos aos campos do modelo.
        model._weightsEmbeddingId = tensorIds.GetProperty("WeightsEmbedding").GetString()!;
        model._weightsInputForgetId = tensorIds.GetProperty("WeightsInputForget").GetString()!;
        model._weightsHiddenForgetId = tensorIds.GetProperty("WeightsHiddenForget").GetString()!;
        model._weightsInputInputId = tensorIds.GetProperty("WeightsInputInput").GetString()!;
        model._weightsHiddenInputId = tensorIds.GetProperty("WeightsHiddenInput").GetString()!;
        model._weightsInputCellId = tensorIds.GetProperty("WeightsInputCell").GetString()!;
        model._weightsHiddenCellId = tensorIds.GetProperty("WeightsHiddenCell").GetString()!;
        model._weightsInputOutputId = tensorIds.GetProperty("WeightsInputOutput").GetString()!;
        model._weightsHiddenOutputId = tensorIds.GetProperty("WeightsHiddenOutput").GetString()!;
        model._biasForgetId = tensorIds.GetProperty("BiasForget").GetString()!;
        model._biasInputId = tensorIds.GetProperty("BiasInput").GetString()!;
        model._biasCellId = tensorIds.GetProperty("BiasCell").GetString()!;
        model._biasOutputId = tensorIds.GetProperty("BiasOutput").GetString()!;
        model._weightsHiddenOutputFinalId = tensorIds.GetProperty("WeightsOutputFinal").GetString()!;
        model._biasOutputFinalId = tensorIds.GetProperty("BiasOutputFinal").GetString()!;
        model._hiddenStateId = tensorIds.GetProperty("HiddenState").GetString()!;
        model._cellStateId = tensorIds.GetProperty("CellState").GetString()!;
        
        // Garante que os pesos de Layer Normalization também sejam carregados.
        model._lnForgetGammaId = tensorIds.GetProperty("LN_Forget_Gamma").GetString()!;
        model._lnForgetBetaId = tensorIds.GetProperty("LN_Forget_Beta").GetString()!;
        model._lnInputGammaId = tensorIds.GetProperty("LN_Input_Gamma").GetString()!;
        model._lnInputBetaId = tensorIds.GetProperty("LN_Input_Beta").GetString()!;
        model._lnCellGammaId = tensorIds.GetProperty("LN_Cell_Gamma").GetString()!;
        model._lnCellBetaId = tensorIds.GetProperty("LN_Cell_Beta").GetString()!;
        model._lnOutputGammaId = tensorIds.GetProperty("LN_Output_Gamma").GetString()!;
        model._lnOutputBetaId = tensorIds.GetProperty("LN_Output_Beta").GetString()!;

        return model;
    }
    catch (Exception ex)
    {
        Console.WriteLine($"[LoadModel] Erro crítico ao carregar o modelo de '{filePath}': {ex.Message}");
        return null;
    }
}

        public void ResetOptimizerState() => _adamOptimizer.Reset();
        public void ClearEpochTemporaryTensors() => _swapManager.ClearAllSwap();
        public IndividualFileTensorManager GetTensorManager() => _tensorManager;
        public DiskSwapManager GetSwapManager() => _swapManager;
        public string GetWeightsEmbeddingId() => _weightsEmbeddingId;
        public string GetWeightsInputForgetId() => _weightsInputForgetId;
        public string GetWeightsHiddenForgetId() => _weightsHiddenForgetId;
        public string GetBiasForgetId() => _biasForgetId;
        public string GetWeightsInputInputId() => _weightsInputInputId;
        public string GetWeightsHiddenInputId() => _weightsHiddenInputId;
        public string GetBiasInputId() => _biasInputId;
        public string GetWeightsInputCellId() => _weightsInputCellId;
        public string GetWeightsHiddenCellId() => _weightsHiddenCellId;
        public string GetBiasCellId() => _biasCellId;
        public string GetWeightsInputOutputId() => _weightsInputOutputId;
        public string GetWeightsHiddenOutputId() => _weightsHiddenOutputId;
        public string GetBiasOutputId() => _biasOutputId;
        public string GetWeightsHiddenOutputFinalId() => _weightsHiddenOutputFinalId;
        public string GetBiasOutputFinalId() => _biasOutputFinalId;
        public string GetLnForgetGammaId() => _lnForgetGammaId;
        public string GetLnForgetBetaId() => _lnForgetBetaId;
        public string GetLnInputGammaId() => _lnInputGammaId;
        public string GetLnInputBetaId() => _lnInputBetaId;
        public string GetLnCellGammaId() => _lnCellGammaId;
        public string GetLnCellBetaId() => _lnCellBetaId;
        public string GetLnOutputGammaId() => _lnOutputGammaId;
        public string GetLnOutputBetaId() => _lnOutputBetaId;

        public void Dispose()
        {
            if (_disposed) return;
            _swapManager?.Dispose();
            _tensorManager?.Dispose();
            _adamOptimizer?.Reset();
            _disposed = true;
            GC.SuppressFinalize(this);
        }
    }
}
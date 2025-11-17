using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Galileu.Node.Core;
using Galileu.Node.Interfaces;

namespace Galileu.Node.Brain
{
    /// <summary>
    /// Estende a rede LSTM base, adaptada para a arquitetura Híbrida 2.0 com cache em disco.
    /// </summary>
    public class GenerativeNeuralNetworkLSTM : NeuralNetworkLSTM
    {
        public readonly VocabularyManager vocabularyManager;
        private readonly ISearchService searchService;
        private readonly int _embeddingSize;
        public int warmupSteps;

        public GenerativeNeuralNetworkLSTM(int vocabSize, int embeddingSize, int hiddenSize, string datasetPath,
            ISearchService? searchService, IMathEngine mathEngine)
            : base(vocabSize, embeddingSize, hiddenSize, vocabSize, mathEngine)
        {
            this.vocabularyManager = new VocabularyManager();
            this.searchService = searchService ?? new MockSearchService();
            this._embeddingSize = embeddingSize;

            int loadedVocabSize = vocabularyManager.BuildVocabulary(datasetPath, maxVocabSize: vocabSize);
            if (loadedVocabSize == 0)
            {
                throw new InvalidOperationException("Vocabulário vazio. Verifique o arquivo de dataset.");
            }
        }

        private GenerativeNeuralNetworkLSTM(NeuralNetworkLSTM baseModel,
            VocabularyManager vocabManager, ISearchService? searchService)
            : base(baseModel)
        {
            if (baseModel == null)
                throw new ArgumentNullException(nameof(baseModel), "Modelo base não pode ser nulo");

            this.vocabularyManager = vocabManager ?? throw new ArgumentNullException(nameof(vocabManager));
            this.searchService = searchService ?? new MockSearchService();

            if (_tensorManager == null || string.IsNullOrEmpty(_weightsEmbeddingId))
            {
                throw new InvalidOperationException("Modelo base está em estado inválido.");
            }

            try
            {
                var shape = _tensorManager.GetShape(_weightsEmbeddingId);
                if (shape == null || shape.Length < 2)
                {
                    throw new InvalidOperationException(
                        $"Shape do embedding inválido: {(shape == null ? "null" : $"[{string.Join(", ", shape)}]")}");
                }

                this._embeddingSize = shape[1];
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Falha ao inicializar GenerativeNeuralNetworkLSTM: {ex.Message}",
                    ex);
            }
        }
        
        

        public static GenerativeNeuralNetworkLSTM? Load(string modelPath, IMathEngine mathEngine,
            VocabularyManager vocabManager, ISearchService? searchService)
        {
            var baseModel = NeuralNetworkLSTM.LoadModel(modelPath, mathEngine);
            if (baseModel == null)
            {
                return null;
            }

            return new GenerativeNeuralNetworkLSTM(baseModel, vocabManager, searchService);
        }

        public string GenerateResponse(string inputText, int maxLength = 50)
        {
            if (string.IsNullOrEmpty(inputText)) return "Erro: Entrada vazia ou nula.";
            return "Geração de resposta não implementada nesta fase.";
        }

        /// <summary>
        /// Calcula a perda para uma sequência usando a arquitetura Híbrida 2.0 com cache em disco.
        /// </summary>
        public float CalculateSequenceLoss(int[] inputIndices, int[] targetIndices)
        {
            var allTempFiles = new List<string>();
            try
            {
                // ✅ CORREÇÃO: Criação do objeto ModelWeights com 'using' para garantir o descarte dos tensores de peso.
                using (var weights = new ModelWeights(this, _mathEngine, _tensorManager))
                {
                    // A chamada para a classe base agora usa os pesos carregados.
                    var (loss, _, generatedFiles) = base.ForwardPassWithOffloading(inputIndices, targetIndices, weights);
                    allTempFiles.AddRange(generatedFiles);
                    return loss;
                }
            }
            finally
            {
                // Garante a limpeza dos arquivos de cache de ativação.
                foreach (var fileId in allTempFiles)
                {
                    _swapManager.DeleteSwapFile(fileId);
                }
            }
        }

        public void Reset() => base.ResetHiddenState();

        private int GetTokenIndex(string token) =>
            vocabularyManager.Vocab.TryGetValue(token.ToLower(), out int tokenIndex)
                ? tokenIndex
                : vocabularyManager.Vocab["<UNK>"];

        private string[] Tokenize(string text) =>
            text.ToLower().Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);

        /// <summary>
        /// ✅ ATUALIZADO E CORRIGIDO: Executa uma verificação de sanidade completa usando a arquitetura Híbrida 2.0.
        /// </summary>
        public void RunSanityCheck()
{
    Console.WriteLine("\n╔═══════════════════════════════════════════════════════════╗");
    Console.WriteLine("║    🚀 INICIANDO VERIFICAÇÃO DE SANIDADE (FINAL)          ║");
    Console.WriteLine("╚═══════════════════════════════════════════════════════════╝");

    var inputIndices = new[] { 5, 10 };
    var targetIndices = new[] { 10, 15 };
    Console.WriteLine($"[Sanity Check] Usando dados sintéticos: Input={{{string.Join(",", inputIndices)}}}, Target={{{string.Join(",", targetIndices)}}}");

    var allTempFiles = new List<string>();
    try
    {
        // ✅ CORREÇÃO: O objeto ModelWeights é criado com 'using' para carregar todos os pesos
        // e garantir sua liberação no final do teste.
        using (var weights = new ModelWeights(this, _mathEngine, _tensorManager))
        {
            // FASE 1: FORWARD PASS
            Console.WriteLine("\n[Sanity Check] Fase 1/3: Executando Forward Pass...");
            var (loss, forwardCache, forwardFiles) = ForwardPassWithOffloading(inputIndices, targetIndices, weights);
            allTempFiles.AddRange(forwardFiles);
            Console.WriteLine($"[Sanity Check] Forward Pass concluído. Perda inicial: {loss:F4}");

            if (float.IsNaN(loss) || float.IsInfinity(loss))
                throw new InvalidOperationException($"Falha na verificação: A perda inicial é {loss}.");
            float expectedLoss = MathF.Log(this.outputSize);
            Console.WriteLine($"[Sanity Check] Perda esperada (aleatória): ~{expectedLoss:F4}");

            // FASE 2: BACKWARD PASS
            Console.WriteLine("\n[Sanity Check] Fase 2/3: Executando Backward Pass...");
            var (gradIds, gradFiles) = BackwardPassWithOffloading(inputIndices, targetIndices, forwardCache, weights);
            allTempFiles.AddRange(gradFiles);
            Console.WriteLine($"[Sanity Check] Backward Pass concluído. {gradIds.Count} arquivos de gradiente gerados.");

            double totalGradSum = 0;
            foreach (var gradId in gradIds.Values)
            {
                using var gradScope = new TensorScope("GradCheck", _mathEngine, _tensorManager);
                var gradTensor = gradScope.LoadTensor(gradId);
                using var gradCpu = gradTensor.ToCpuTensor();
                foreach (var val in gradCpu.GetData())
                {
                    if (float.IsNaN(val) || float.IsInfinity(val))
                        throw new InvalidOperationException($"Falha na verificação: Gradiente contém valor inválido ({val}).");
                    totalGradSum += Math.Abs(val);
                }
            }
            Console.WriteLine($"[Sanity Check] Soma absoluta de todos os gradientes: {totalGradSum:E2}");
            if (totalGradSum < 1e-9)
                throw new InvalidOperationException("Falha na verificação: A soma dos gradientes é próxima de zero.");

            // FASE 3: UPDATE PASS
            Console.WriteLine("\n[Sanity Check] Fase 3/3: Executando Update Pass (Adam)...");
            var weightIds = new Dictionary<string, string> { { "W_hy", _weightsHiddenOutputFinalId }, { "B_y", _biasOutputFinalId } };
            UpdateAdamGPUPassZeroRAM(weightIds, gradIds);
            Console.WriteLine("[Sanity Check] Update Pass concluído.");
        }

        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine("\n╔═══════════════════════════════════════════════════════════╗");
        Console.WriteLine("║         ✅ VERIFICAÇÃO DE SANIDADE CONCLUÍDA COM SUCESSO!      ║");
        Console.WriteLine("╚═══════════════════════════════════════════════════════════╝\n");
    }
    catch (Exception ex)
    {
        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine("\n╔═══════════════════════════════════════════════════════════╗");
        Console.WriteLine("║         ❌ FALHA NA VERIFICAÇÃO DE SANIDADE!               ║");
        Console.WriteLine("╚═══════════════════════════════════════════════════════════╝");
        Console.WriteLine($"[Sanity Check] ERRO: {ex.Message}");
        Console.ResetColor();
        throw;
    }
    finally
    {
        Console.WriteLine("\n[Sanity Check] Executando limpeza de recursos...");
        foreach (var fileId in allTempFiles)
        {
            _swapManager.DeleteSwapFile(fileId);
            _tensorManager.DeleteTensor(fileId);
        }
        Console.WriteLine("[Sanity Check] Limpeza concluída.");
    }
}

        public IMathEngine GetMathEngine() => _mathEngine;
    }
}
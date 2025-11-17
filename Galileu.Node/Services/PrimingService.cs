using Galileu.Node.Brain;
using Galileu.Node.Core;
using Galileu.Node.Interfaces;
using System;
using System.IO;
using System.Linq;

namespace Galileu.Node.Services
{
    public class PrimingService
    {
        private readonly string _promptFilePath;

        public PrimingService()
        {
            _promptFilePath = Path.Combine(Environment.CurrentDirectory, "Dayson", "priming_prompt.txt");
        }

        /// <summary>
        /// Processa o prompt de diretiva para "aquecer" o estado oculto do modelo,
        /// usando a arquitetura final de Memória Gerenciada.
        /// </summary>
        public void PrimeModel(GenerativeNeuralNetworkLSTM model)
        {
            if (!File.Exists(_promptFilePath))
            {
                Console.WriteLine($"[PrimingService] Aviso: Arquivo de prompt não encontrado em '{_promptFilePath}'.");
                return;
            }

            Console.WriteLine("[PrimingService] Inicializando o modelo com a diretiva de comportamento...");

            var promptText = File.ReadAllText(_promptFilePath);
            if (string.IsNullOrWhiteSpace(promptText))
            {
                Console.WriteLine("[PrimingService] Aviso: Arquivo de prompt está vazio.");
                return;
            }

            var vocabManager = model.vocabularyManager;

            var tokens = promptText.ToLower().Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            var inputIndices = tokens
                .Select(token => vocabManager.Vocab.TryGetValue(token, out int index) ? index : vocabManager.Vocab["<UNK>"])
                .ToArray();

            var targetIndices = new int[inputIndices.Length]; 

            try
            {
                // ✅ CORREÇÃO: O objeto ModelWeights agora é IDisposable e deve ser criado com 'using'.
                // Ele carrega todos os pesos do modelo na VRAM e garante sua liberação no final.
                using (var weights = new ModelWeights(model, model.GetMathEngine(), model.GetTensorManager()))
                {
                    // A chamada para RunForwardPassForInference usa os pesos carregados.
                    // O gerenciamento de memória dos tensores de ativação é feito internamente.
                    model.RunForwardPassForInference(inputIndices, targetIndices, weights);
                }

                Console.WriteLine("[PrimingService] Modelo inicializado com sucesso.");
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[PrimingService] ERRO CRÍTICO durante a inicialização do modelo: {ex.Message}");
                Console.ResetColor();
            }
        }
    }
}
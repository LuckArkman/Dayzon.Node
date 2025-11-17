using Galileu.Node.Interfaces;
using System;
using System.Collections.Generic;

namespace Galileu.Node.Brain
{
    /// <summary>
    /// Contêiner para os tensores de peso do modelo.
    /// Implementa IDisposable para garantir que todos os tensores carregados na VRAM
    /// para uma época sejam liberados de forma determinística no final.
    /// </summary>
    public class ModelWeights : IDisposable
    {
        public ModelWeights(){}
        private readonly List<IMathTensor> _loadedTensors = new List<IMathTensor>();
        private bool _disposed = false;

        // Propriedades para cada peso do modelo
        public IMathTensor Embedding { get; }
        public IMathTensor W_if { get; }
        public IMathTensor W_hf { get; }
        public IMathTensor B_f { get; }
        // ... (adicione todas as outras propriedades de peso aqui)
        public IMathTensor W_ii { get; }
        public IMathTensor W_hi { get; }
        public IMathTensor B_i { get; }
        public IMathTensor W_ic { get; }
        public IMathTensor W_hc { get; }
        public IMathTensor B_c { get; }
        public IMathTensor W_io { get; }
        public IMathTensor W_ho { get; }
        public IMathTensor B_o { get; }
        public IMathTensor W_hy { get; }
        public IMathTensor B_y { get; }
        public IMathTensor LN_f_gamma { get; }
        public IMathTensor LN_f_beta { get; }
        public IMathTensor LN_i_gamma { get; }
        public IMathTensor LN_i_beta { get; }
        public IMathTensor LN_c_gamma { get; }
        public IMathTensor LN_c_beta { get; }
        public IMathTensor LN_o_gamma { get; }
        public IMathTensor LN_o_beta { get; }


        public ModelWeights(NeuralNetworkLSTM model, IMathEngine mathEngine, IndividualFileTensorManager tensorManager)
        {
            // O construtor carrega todos os tensores do disco e os rastreia para descarte posterior.
            Embedding = Track(tensorManager.LoadTensor(model.GetWeightsEmbeddingId()));
            W_if = Track(tensorManager.LoadTensor(model.GetWeightsInputForgetId()));
            W_hf = Track(tensorManager.LoadTensor(model.GetWeightsHiddenForgetId()));
            B_f = Track(tensorManager.LoadTensor(model.GetBiasForgetId()));
            // ... (carregue todos os outros pesos aqui da mesma forma)
            W_ii = Track(tensorManager.LoadTensor(model.GetWeightsInputInputId()));
            W_hi = Track(tensorManager.LoadTensor(model.GetWeightsHiddenInputId()));
            B_i = Track(tensorManager.LoadTensor(model.GetBiasInputId()));
            W_ic = Track(tensorManager.LoadTensor(model.GetWeightsInputCellId()));
            W_hc = Track(tensorManager.LoadTensor(model.GetWeightsHiddenCellId()));
            B_c = Track(tensorManager.LoadTensor(model.GetBiasCellId()));
            W_io = Track(tensorManager.LoadTensor(model.GetWeightsInputOutputId()));
            W_ho = Track(tensorManager.LoadTensor(model.GetWeightsHiddenOutputId()));
            B_o = Track(tensorManager.LoadTensor(model.GetBiasOutputId()));
            W_hy = Track(tensorManager.LoadTensor(model.GetWeightsHiddenOutputFinalId()));
            B_y = Track(tensorManager.LoadTensor(model.GetBiasOutputFinalId()));
            LN_f_gamma = Track(tensorManager.LoadTensor(model.GetLnForgetGammaId()));
            LN_f_beta = Track(tensorManager.LoadTensor(model.GetLnForgetBetaId()));
            LN_i_gamma = Track(tensorManager.LoadTensor(model.GetLnInputGammaId()));
            LN_i_beta = Track(tensorManager.LoadTensor(model.GetLnInputBetaId()));
            LN_c_gamma = Track(tensorManager.LoadTensor(model.GetLnCellGammaId()));
            LN_c_beta = Track(tensorManager.LoadTensor(model.GetLnCellBetaId()));
            LN_o_gamma = Track(tensorManager.LoadTensor(model.GetLnOutputGammaId()));
            LN_o_beta = Track(tensorManager.LoadTensor(model.GetLnOutputBetaId()));

        }

        private IMathTensor Track(IMathTensor tensor)
        {
            _loadedTensors.Add(tensor);
            return tensor;
        }

        public void Dispose()
        {
            if (_disposed) return;
            foreach (var tensor in _loadedTensors)
            {
                tensor.Dispose();
            }
            _loadedTensors.Clear();
            _disposed = true;
        }
    }
}
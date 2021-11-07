from modules.swing_classifier.model.lstm_model import LSTMModel


def get_model(config):
    """Get Model Function"""
    
    model_name = config['model']['name']
    if model_name == 'SwingClassifier':
        return LSTMModel(config)
    else:
        NotImplementedError('The model is not supported.')
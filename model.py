# HardGNN Model - Standalone Version  
# This file imports the full-featured HardGNN model from HardGNN_model.py
# and provides backward compatibility with the original interface

from HardGNN_model import Recommender

# Alias for backward compatibility
class HardGNNRecommender(Recommender):
    """
    Backward compatible alias for the Recommender class.
    This allows existing code to work with both class names.
    """
    pass

# Also support the original Recommender name for flexibility 
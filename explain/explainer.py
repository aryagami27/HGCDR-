
class RecommendationExplainer:
    """
    Provides attribution and explanations for recommendations.
    """
    def explain(self, user_id, item_id, signals):
        """
        Attributes the recommendation score to various model components.
        
        Args:
            user_id: ID of the user (int or str).
            item_id: ID of the recommended item (int or str).
            signals: Dictionary containing intermediate model outputs or attention weights.
                     Expected keys: 'transfer_weight', 'kg_path', 'history_overlap', etc.
                     
        Returns:
            Dictionary with structured explanation values.
        """
        explanation = {
            "user_id": user_id,
            "item_id": item_id
        }
        
        # 1. Cross-Domain Transfer Attribution
        # How much did the source domain contribute?
        if "transfer_weight" in signals:
            # Assuming transfer_weight is a float or 0-d tensor
            val = signals["transfer_weight"]
            if hasattr(val, 'item'):
                val = val.item()
            explanation["cross_domain_contribution"] = val
            
        # 2. KG Path Attribution
        # Which paths in the knowledge graph connected the user to the item?
        if "kg_path" in signals:
            explanation["kg_evidence"] = signals["kg_path"]
            
        # 3. User History Attribution
        # Did this item match specific items in user history?
        if "history_overlap" in signals:
             explanation["history_match_score"] = signals["history_overlap"]
             
        return explanation

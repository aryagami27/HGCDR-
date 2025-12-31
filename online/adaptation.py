
import torch

def adapt_user(user_model, support_data, lr=1e-3, steps=3):
    """
    Adapts the user model (or specific user parameters) to new data 
    to mitigate drift.
    
    Args:
        user_model: PyTorch nn.Module (the user encoder or specific user layers).
        support_data: Tuple or dict containing recent interaction data (x, y, etc.).
        lr: Learning rate for the adaptation step.
        steps: Number of gradient steps to take.
    
    Returns:
        The updated model (in-place modification usually, but explicit return for clarity).
    """
    # Create a local optimizer for the adaptation steps
    # We only optimize the parameters of the user_model passed in.
    optimizer = torch.optim.Adam(user_model.parameters(), lr=lr)
    
    # Assuming user_model has a method 'compute_loss' or we use standard logic.
    # To keep it generic based on the prompt skeleton:
    # "loss = user_model.compute_loss(support_data)"
    
    user_model.train()
    
    for _ in range(steps):
        # We assume support_data can be unpacked or passed directly.
        # This part relies on the specific interface of the user_model.
        # If compute_loss is not available, this will raise an AttributeError,
        # forcing the user to implement it on their model class.
        if hasattr(user_model, 'compute_loss'):
            loss = user_model.compute_loss(support_data)
        else:
            # Fallback for generic pytorch modules if support_data is (inputs, targets)
            # This is a bit speculative without the full model code, but safe for now
            # as the prompt implies we just need to write the loop structure.
            # Let's assume the user handles the model interface.
            raise NotImplementedError("User model must implement 'compute_loss(data)' for adaptation.")
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return user_model

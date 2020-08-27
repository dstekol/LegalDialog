from transformers import AdamW, get_linear_schedule_with_warmup

def create_optimizer(model, weight_decay, lr, epsilon):
    """ Creates an AdamW optimizer with the given config options. """
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=epsilon)
    return optimizer

def create_scheduler(optimizer, warmup_steps, total_steps):
    """ Creates an learning rate scheduler with the given config options. """
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    return scheduler

def step_model(model, loss, retain_graph, writer, writer_tag):
    """ Performs a weight update for the given model, and logs the current loss.
        The retain_graph parameter can be set to true if future passes through the gradient graph will be needed."""
    # logging
    writer.add_scalar(writer_tag, loss.item(), model.step)
    
    # backpropagation
    model.optimizer.zero_grad()
    loss.backward(retain_graph=retain_graph)

    # updates
    model.optimizer.step()
    model.scheduler.step()
    model.step += 1

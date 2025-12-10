import os 
import torch 
from tqdm import tqdm

def train_disentangled_clip(model,train_loader,optimizer,writer,output_directory,num_epochs,
    device,loss_fn,save_every=10,):
    
    best_loss = float("inf")
    best_model_dir = None
    global_step = 0

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            loss = loss_fn(outputs["vision_embeds"], outputs["text_embeds"])
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            writer.add_scalar("Loss/train", loss.item(), global_step)
            epoch_loss += loss.item()
            global_step += 1
            progress_bar.set_postfix({"batch_loss": loss.item()})

        if (epoch + 1) % save_every == 0:
            step_save_dir = os.path.join(output_directory, f"epoch_{epoch+1}")
            model.save_from_pretrained(step_save_dir)

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_dir = os.path.join(output_directory, "best_model")
            model.save_from_pretrained(best_model_dir)

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Average loss: {avg_epoch_loss:.4f}")

    print(f"Best model saved at {best_model_dir} with loss {best_loss:.4f}")
    writer.close()
    return best_model_dir, best_loss
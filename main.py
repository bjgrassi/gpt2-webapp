from flask import Flask, render_template, request
import torch

app = Flask(__name__)

# Initialize models as None
models = {
    "pretrained": None,
    "finetuned": None,
    "early_exit": None
}

def get_models():
    if not hasattr(get_models, 'loaded_models'):
        print("\n⭐️ INITIALIZING MODELS (This should happen ONCE) ⭐️")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Lazy imports - only import when needed
        from models.pretrained_gpt2 import GPT as PretrainedGPT, generate_text as pretrained_generate
        from models.finetuned_gpt2 import GPT as FinetunedGPT, generate_text as finetuned_generate
        from models.early_exit_gpt2 import GPT as EarlyExitGPT, generate_text as early_exit_generate
        
        models["pretrained"] = PretrainedGPT.from_pretrained("gpt2").to(device)
        models["finetuned"] = FinetunedGPT.from_pretrained("gpt2").to(device)
        models["early_exit"] = EarlyExitGPT.from_pretrained("gpt2").to(device)
        
        # Store the generate functions too
        models["_generate_functions"] = {
            "pretrained": pretrained_generate,
            "finetuned": finetuned_generate,
            "early_exit": early_exit_generate
        }
        
        print("✅ ALL MODELS LOADED SUCCESSFULLY!\n")
        get_models.loaded_models = models
    
    return get_models.loaded_models

@app.route('/', methods=['GET', 'POST'])
def index():
    models = get_models()
    
    if request.method == 'POST':
        prompt = request.form['prompt']
        model_type = request.form['model_type']
        
        try:
            generate_func = models["_generate_functions"].get(model_type)
            if generate_func:
                generated_text = generate_func(models[model_type], prompt)
            else:
                generated_text = "Invalid model selection"
                
            return render_template('result.html', 
                                prompt=prompt, 
                                generated_text=generated_text, 
                                model_type=model_type)
        except Exception as e:
            return f"Error: {str(e)}"
    
    return render_template('index.html')

if __name__ == '__main__':
    print("\n🔥 STARTING FLASK APPLICATION 🔥")
    app.run(debug=True)
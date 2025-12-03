from email import utils
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import brighteyes_ism.analysis.Tools_lib as tools
import brighteyes_ism.analysis.Graph_lib as gra
from IPython.display import display, HTML
from matplotlib.patches import Circle
from PIL import Image
from microssim import MicroSSIM, micro_structural_similarity

from varreg import varreg, loss, utils
from varreg import run_and_save as rs
from s2ism import s2ism as s2


def frequency_axes(shape, pxsize: float = 1):
    sy = shape[-3]
    sx = shape[-2]
 
    fx =  np.fft.fftshift(np.fft.fftfreq(sx, pxsize))
    fy =  np.fft.fftshift(np.fft.fftfreq(sy, pxsize))
 
    return [fy, fx]

def save_gif(tensore, params, indici, filepath, min_val=0, max_val=1e7, cmap='viridis', fps=5):
    frames = []
    _, Nx, Ny = tensore.shape
    cx, cy = 0, 0  #centro del cerchio
    radius = params["cutoff"] #raggio del cerchio
    kspace = frequency_axes(tensore.shape, params["pixelsize"])
    kmapx, kmapy = np.meshgrid(kspace[0],kspace[1])

    for i in indici:
        fig, ax = plt.subplots()
        im = ax.imshow(tensore[i], vmin=min_val, vmax=max_val, cmap=cmap, extent = [kmapx.min(), kmapx.max(), kmapy.min(),kmapy.max()])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        #Add a cutout circle
        circle = Circle((cy, cx), radius, edgecolor='cyan', facecolor='none',
                         linestyle='dashed', linewidth=1.5)
        ax.add_patch(circle)
        ax.axis('off')
        ax.set_title(f"K={params['K']}, n_iter={i}, lambda={params['lambda']}, W={params['W']}, method={params['method']}", fontsize=16)
        
        fig.tight_layout(pad=0.1)
        fig.canvas.draw()
        img_rgba = np.asarray(fig.canvas.buffer_rgba())
        img_rgb = Image.fromarray(img_rgba).convert("RGB")
        frames.append(np.array(img_rgb))
        
        plt.close(fig)
        
    imageio.mimsave(filepath, frames, fps=fps)

def create_entry_dataframe(index, params, fig_paths):
    return {
        "id": index,
        "K": params["K"],
        "signal": params["signal"],
        "num_iter": params["num_iter"],
        "lambda": params["lambda"],
        "alpha": params["alpha"],
        "method": params["method"],
        "regularizer": params["regularizer"],
        "initialization": params["initialization"],
        "W": params["W"],
        "lr": params["lr"],
        "Figure results": f"<a href='{fig_paths[0]}' target='_blank'>Figure results</a>",
        "Figure loss": f"<a href='{fig_paths[1]}' target='_blank'>Figure loss</a>",
        "Figure KL": f"<a href='{fig_paths[2]}' target='_blank'>Figure KL</a>",
        "Figure best iter": f"<a href='{fig_paths[3]}' target='_blank'>Figure best iter</a>",
        "Figure PSNR": f"<a href='{fig_paths[4]}' target='_blank'>Figure PSNR</a>",
    }

def save_experiment_results(fig_results, fig_loss, fig_KL, fig_best, fig_PSNR, params, output_dir, print_csv=True):

    """
    Save the results of an experiment to a file.
    
    Parameters:
        fig_results: Figure for the results.
        fig_loss: Figure for the loss over iterations.
        fig_KL: Figure for the KL divergence over iterations.
        fig_best: Figure for the best iteration result.
        fig_PSNR: Figure for the PSNR over iterations.
        params: Parameters of the experiment.
        output_dir: Output directory for saving the results.
        save_gif: Whether to save GIFs.
        print_csv: Whether to print the CSV log.

    """

    log_path = os.path.join(output_dir, "esperiments_log.csv")

    if os.path.exists(log_path):
        df_log = pd.read_csv(log_path)
        next_id = df_log["id"].max() + 1
    else:
        df_log = pd.DataFrame()
        next_id = 0

    # Create paths for figures and GIFs
    fig1_path = os.path.join(output_dir, f"{next_id}_figResults.png")
    fig2_path = os.path.join(output_dir, f"{next_id}_figLoss.png")
    fig3_path = os.path.join(output_dir, f"{next_id}_figKL.png")
    fig4_path = os.path.join(output_dir, f"{next_id}_figBestIter.png")
    fig5_path = os.path.join(output_dir, f"{next_id}_figPSNR.png")


    # Save figures
    fig_results.savefig(fig1_path)
    fig_loss.savefig(fig2_path)
    fig_KL.savefig(fig3_path)
    fig_best.savefig(fig4_path)
    fig_PSNR.savefig(fig5_path)

    # New row
    new_entry = create_entry_dataframe(next_id, params, [fig1_path, fig2_path, fig3_path, fig4_path, fig5_path])
    df_log = pd.concat([df_log, pd.DataFrame([new_entry])], ignore_index=True)
    df_log.to_csv(log_path, index=False)

    df_log.style.set_properties(**{'text-align': 'left'})
    

    # Visualizza HTML con link cliccabili
    if print_csv:
        display(HTML(df_log.to_html(escape=False)))

def run_single_experiment(dset, psf, params, weight, ground_truth):
    """
    Run a single experiment with the given parameters and save the results.
    
    Parameters:
        dset (ndarray): Input dataset of shape Nx x Ny x K.
        psf (ndarray): Point spread function of shape Mx x My.
        params (dict): Dictionary of parameters for the experiment.
        weight (ndarray): Weight map of shape Nx x Ny.
        ground_truth (ndarray): Ground truth image of shape Nx x Ny.
        
    Returns:
        dict: Dictionary containing the results of the experiment.
    """

    # Run the experiment
    mid_sol, *_ = s2.max_likelihood_reconstruction(dset.sum(axis=-1), psf[np.newaxis,...], max_iter = params["num_iter"], rep_to_save ='all')

    ISM_SGM_solution, _ = varreg.variance_reconstruction(dset, psf, lambda_value=params["lambda"], alpha_value=params["alpha"], weight_map=weight,
                            max_iter=params["num_iter"], tol=None, method=params["method"],
                            regularizer=params["regularizer"], initialization=params["initialization"], normalize=True,
                            process='gpu', rec_to_save='all', binomial = False)
    ISM_SGM_mean = ISM_SGM_solution.mean(axis=-1)
    ISM_SGM_norm = torch.round(ISM_SGM_mean * (dset.sum() / ISM_SGM_mean.sum((1,2))[:,None,None])).int()
    
    ISM_SGP_solution, step_size = varreg.variance_reconstruction(dset, psf, lambda_value=params["lambda"], alpha_value=params["alpha"], weight_map=weight,
                            max_iter=params["num_iter"], tol=None, method=params["method"],
                            regularizer=params["regularizer"], initialization=params["initialization"], normalize=True,
                            process='gpu', rec_to_save='all', binomial = False, step_size_search=params["step_size_search"],
                            line_search_params=params["line_search"], verbose=False)
    step_size_array = torch.stack(step_size).cpu().numpy()
    ISM_SGP_norm = ISM_SGP_solution.sum(axis=-1)
    
    # KL divergence computation
    kl_mid = utils.kl_divergence(ground_truth, mid_sol, normalize_entries = True)
    kl_SGM = utils.kl_divergence(ground_truth, ISM_SGM_norm.numpy().astype(np.float64), normalize_entries = True)
    kl_SGP = utils.kl_divergence(ground_truth, ISM_SGP_norm.numpy().astype(np.float64), normalize_entries = True)
    
    # Compute SSIM
    SSIM_mid = np.zeros(params["num_iter"])
    SSIM_SGM = np.zeros(params["num_iter"])
    SSIM_SGP = np.zeros(params["num_iter"])

    for i in range(params["num_iter"]):
        SSIM_mid[i] = micro_structural_similarity(ground_truth, mid_sol[i])
        SSIM_SGM[i] = micro_structural_similarity(ground_truth, ISM_SGM_norm[i].numpy().astype(np.float64))
        SSIM_SGP[i] = micro_structural_similarity(ground_truth, ISM_SGP_norm[i].numpy().astype(np.float64))
    
    # Compute PSNR 
    PSNR_mid = utils.peak_signal_to_noise_ratio(ground_truth, mid_sol)
    PSNR_SGM = utils.peak_signal_to_noise_ratio(ground_truth, ISM_SGM_norm.numpy().astype(np.float64))
    PSNR_SGP = utils.peak_signal_to_noise_ratio(ground_truth, ISM_SGP_norm.numpy().astype(np.float64))
            
    # Loss computation
    full_KL = False
    fidelity_mid = loss.fidelity_loss(torch.tensor(dset.sum(-1)), torch.tensor(mid_sol), torch.tensor(psf), full_KL=full_KL, k_loss=False)
    fidelity_sgm = loss.fidelity_loss(torch.tensor(dset), ISM_SGM_solution, torch.tensor(psf), full_KL=full_KL, k_loss=True)
    fidelity_sgp = loss.fidelity_loss(torch.tensor(dset), ISM_SGP_solution, torch.tensor(psf), full_KL=full_KL, k_loss=True)
        
    regularization_sgm = loss.regularization_loss(ISM_SGM_solution, iteration_axis=True).numpy()
    regularization_sgp = loss.regularization_loss(ISM_SGP_solution, iteration_axis=True).numpy()
    
    fidelity_loss_sgm = params["alpha"]*fidelity_sgm
    reg_loss_sgm = params["lambda"]*regularization_sgm
    total_loss_sgm = fidelity_loss_sgm + reg_loss_sgm

    fidelity_loss_sgp = params["alpha"]*fidelity_sgp
    reg_loss_sgp = params["lambda"]*regularization_sgp
    total_loss_sgp = fidelity_loss_sgp + reg_loss_sgp
    
    mid_result = {
        'solution': mid_sol,
        "fidelity": fidelity_mid.numpy(),
        "kl_divergence": kl_mid,
        "SSIM": SSIM_mid,
        "PSNR": PSNR_mid
    }
    
    sgm_result = {
        'k_solutions': ISM_SGM_solution,
        'solution': ISM_SGM_norm.numpy().astype(np.float64),
        "fidelity": fidelity_sgm.numpy(),
        'regularization': regularization_sgm,
        'total_loss': total_loss_sgm.sum(-1),
        "kl_divergence": kl_SGM,
        "SSIM": SSIM_SGM,
        "PSNR": PSNR_SGM
    }
    
    sgp_result = {
        'k_solutions': ISM_SGP_solution,
        'step_size': step_size_array,
        'solution': ISM_SGP_norm.numpy().astype(np.float64),
        "fidelity": fidelity_sgp.numpy(),
        'regularization': regularization_sgp,
        'total_loss': total_loss_sgp.sum(-1),
        "kl_divergence": kl_SGP,
        "SSIM": SSIM_SGP,
        "PSNR": PSNR_SGP
    }
    
    return mid_result, sgm_result, sgp_result
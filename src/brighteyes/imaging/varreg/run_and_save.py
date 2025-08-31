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

from varreg import varreg, loss, utils
from varreg import run_and_save as rs


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

def run_single_experiment(dset, psf, params, weight, ground_truth, img, rl, kl_rl, output_dir, compute_adam=False, plot_K=False, print_csv=False):
    """
    Run a single experiment with the given parameters and save the results.
    
    Parameters:
        dset (ndarray): Input dataset of shape Nx x Ny x K.
        psf (ndarray): Point spread function of shape Mx x My.
        params (dict): Dictionary of parameters for the experiment.
        weight (ndarray): Weight map of shape Nx x Ny.
        ground_truth (ndarray): Ground truth image of shape Nx x Ny.
        img (ndarray): Image of shape Nx x Ny. It is the dset summed over K.
        rl (ndarray): Richardson-Lucy reconstruction of shape num_iter x Nx x Ny.
        kl_rl (ndarray): KL divergence for Richardson-Lucy reconstruction array of length num_iter.
        output_dir (str): Output directory for saving results.
        compute_adam (bool): Whether to compute Adam solution.
        plot_K (bool): Whether to plot the figure of all K solutions.
        print_csv (bool): Whether to print the CSV log.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Run the experiment
    varreg_solution = varreg.variance_reconstruction(dset, psf, lambda_value=params["lambda"], alpha_value=params["alpha"], W=weight,
                            max_iter=params["num_iter"], tol=None, method=params["method"],
                            regularizer=params["regularizer"], initialization=params["initialization"], normalize=False,
                            process='gpu', rec_to_save='all')
    varreg_mean = varreg_solution.mean(axis=-1)
    
    adam_solution, losses = varreg.adam_optimize_obj(dset, psf, num_iter=params["num_iter"], lambda_value=params["lambda"], alpha_value=params["alpha"],
                                                 initialization=params["initialization"], W=weight, regularizer=params["regularizer"],
                                                 lr=params["lr"], process='gpu', full_KL=False, rec_to_save='all', verbose=False, sum_rep = False)
    adam_mean = adam_solution.sum(axis=-1)
    del adam_solution

    pxsizex = params["pixelsize"]
    full_KL = False
    rec_idx=-1
    n_rep = dset.shape[-1]
    psf_rep = np.repeat(psf[..., np.newaxis], n_rep, -1)

    fig_res, ax_res = plt.subplots(1, 5, figsize=(30, 7))
    gra.ShowImg(img[15: -15, 15:-15], pxsizex*1e-3, fig = fig_res, ax=ax_res[0])
    ax_res[0].set_title('Raw Image')
    gra.ShowImg(rl[rec_idx, 15: -15, 15:-15], pxsizex*1e-3, fig = fig_res, ax=ax_res[1])
    ax_res[1].set_title('Richardson-Lucy')
    gra.ShowImg(varreg_mean[rec_idx, 15: -15, 15:-15], pxsizex*1e-3, fig = fig_res, ax=ax_res[2])
    ax_res[2].set_title('Variance Regularization - Mean')
    gra.ShowImg(adam_mean[rec_idx, 15: -15, 15:-15], pxsizex*1e-3, fig = fig_res, ax=ax_res[3])
    ax_res[3].set_title('Adam Optimization')
    gra.ShowImg(ground_truth[15: -15, 15:-15], pxsizex*1e-3, fig = fig_res, ax=ax_res[4])
    ax_res[4].set_title('Ground Truth')
    fig_res.suptitle(f"K={params['K']}, n_iter={params['num_iter'] if rec_idx==-1 else rec_idx}, \
                lambda={params['lambda']}, W={params['W']}, method={params['method']}", fontsize=16)
            
    # KL divergence
    fidelity_rl = loss.fidelity_loss(torch.tensor(img), torch.tensor(rl), torch.tensor(psf))
    fidelity = loss.fidelity_loss(torch.tensor(dset), varreg_solution, torch.tensor(psf_rep))
    regularization = loss.regularization_loss(varreg_solution, iteration_axis=True)
    del varreg_solution

    fidelity_loss = params["alpha"]*fidelity
    reg_loss = params["lambda"]*regularization
    total_loss = fidelity_loss + reg_loss
    
    if compute_adam:
        total_loss_adam = np.asarray(losses['total'])
        fidelity_adam = np.asarray(losses['fidelity'])
        regularization_adam = np.asarray(losses['regularization'])
        learning_rate_adam = np.asarray(losses['learning_rate'])
    
        fig_loss, ax_loss = plt.subplots(2,3, figsize=(20,10))

        ax_loss[0,0].plot(fidelity_rl, color = 'red', label = 'fidelity_rl')
        ax_loss[0,0].legend()
        ax_loss[0,0].set_title('Richardson-Lucy')
        ax_loss[0,0].set_xlabel('Iterations')
        ax_loss[0,0].set_ylabel('Loss')

        ax_loss[0,1].plot(fidelity_loss, label = 'fidelity')
        ax_loss[0,1].plot(total_loss, label = 'total = fidelity + lambda*reg')
        ax_loss[0,1].legend()
        ax_loss[0,1].set_title('Variance Regularization')
        ax_loss[0,1].set_xlabel('Iterations')
        ax_loss[0,1].set_ylim([total_loss.min()*0.95, total_loss[-1]*1.05])

        ax_loss[0,2].plot(reg_loss, color='green', label = 'regularization')
        ax_loss[0,2].legend()
        ax_loss[0,2].set_title('Variance Regularization')
        ax_loss[0,2].set_xlabel('Iterations')
        ax_loss[0,2].set_ylabel('Loss')

        #New line

        ax_loss[1,0].plot(learning_rate_adam, color = 'red', label = 'fidelity_rl')
        ax_loss[1,0].legend()
        ax_loss[1,0].set_title('Learning rate Adam')
        ax_loss[1,0].set_xlabel('Iterations')
        ax_loss[1,0].set_ylabel('lr')

        ax_loss[1,1].plot(fidelity_adam, label = 'fidelity')
        ax_loss[1,1].plot(total_loss_adam, label = 'total = fidelity + lambda*reg')
        ax_loss[1,1].legend()
        ax_loss[1,1].set_title('Variance Regularization')
        ax_loss[1,1].set_xlabel('Iterations')
        #ax_loss[1,1].set_ylim([total_loss_adam.min()*0.95, total_loss_adam[-1]*1.05])

        ax_loss[1,2].plot(regularization_adam, color='green', label = 'regularization')
        ax_loss[1,2].legend()
        ax_loss[1,2].set_title('Variance Regularization')
        ax_loss[1,2].set_xlabel('Iterations')
        ax_loss[1,2].set_ylabel('Loss')
        #ax_loss[1,2].set_ylim([0, regularization_adam.max()*1.05])

        # Added lateral description
        fig_loss.text(0.06, 0.7, 'sgm', va='center', rotation='vertical', fontsize=14, fontweight='bold')
        fig_loss.text(0.06, 0.3, 'Adam', va='center', rotation='vertical', fontsize=14, fontweight='bold')
        
        fig_loss.suptitle(str(params)[1:-1], fontsize=16)

    else:
        fig_loss, ax_loss = plt.subplots(1,3, figsize=(20,10))
        ax_loss[0].plot(fidelity_rl, color = 'red', label = 'learning_rate')
        ax_loss[0].legend()
        ax_loss[0].set_title('Richardson-Lucy')
        ax_loss[0].set_xlabel('Iterations')
        ax_loss[0].set_ylabel('Loss')

        ax_loss[1].plot(fidelity_loss, label = 'fidelity')
        ax_loss[1].plot(total_loss, label = 'total = alpha*fidelity + lambda*reg')
        ax_loss[1].legend()
        ax_loss[1].set_title('Variance Regularization')
        ax_loss[1].set_xlabel('Iterations')
        #ax[1,1].set_ylim([total_loss_adam.min()*0.95, total_loss_adam.max()*1.05])

        ax_loss[2].plot(reg_loss, color='green', label = 'regularization')
        ax_loss[2].legend()
        ax_loss[2].set_title('Variance Regularization')
        ax_loss[2].set_xlabel('Iterations')
        ax_loss[2].set_ylabel('Loss')
        
        fig_loss.suptitle(str(params)[1:-1], fontsize=16)
        
        
    #plot KL divergence
    kl_rl = utils.kl_divergence(ground_truth, rl, normalize_entries = True)
    kl_var = utils.kl_divergence(ground_truth, varreg_mean, normalize_entries = True)
    kl_adam = utils.kl_divergence(ground_truth, adam_mean, normalize_entries = True)

    
    idx_rl = np.argmin(kl_rl)
    idx_varreg = np.argmin(kl_var)
    idx_adam = np.argmin(kl_adam)

    fig_kl, ax_kl = plt.subplots(1, 1, figsize=(20, 10))

    ax_kl.plot(kl_rl, color='r', label = f'Richardson-Lucy KL_min={kl_rl[idx_rl]:.3f}')
    ax_kl.plot(kl_var, color='C2', label = f'Variance Regularization - Mean KL_min={kl_var[idx_varreg]:.3f}')

    ax_kl.scatter(idx_rl, kl_rl[idx_rl], color='r')
    ax_kl.scatter(idx_varreg, kl_var[idx_varreg], color='C2')

    ax_kl.set_xticks(list(ax_kl.get_xticks()[1:]) + [idx_rl])
    ax_kl.set_xticks(list(ax_kl.get_xticks()[1:]) + [idx_varreg])
    
    ax_kl.plot(kl_adam, color='C0', label = f'Adam Optimization KL_min={kl_adam[idx_adam]:.3f}')
    ax_kl.scatter(idx_adam, kl_adam[idx_adam], color='C0')
    ax_kl.set_xticks(list(ax_kl.get_xticks()[1:]) + [idx_adam])
    
    ax_kl.set_xlabel('Iteration')
    ax_kl.set_ylabel('Normalized KL')
    ax_kl.legend()
    ax_kl.set_title(str(params)[1:-1], fontsize=16)
    
    # Plot best iteration
    fig_bestres, ax_bestres = plt.subplots(1, 5, figsize=(30, 7))

    gra.ShowImg(img[15: -15, 15:-15], pxsizex*1e-3, fig = fig_bestres, ax=ax_bestres[0])
    ax_bestres[0].set_title('Raw Image')
    gra.ShowImg(rl[idx_rl, 15: -15, 15:-15], pxsizex*1e-3, fig = fig_bestres, ax=ax_bestres[1])
    ax_bestres[1].set_title('Richardson-Lucy')
    gra.ShowImg(varreg_mean[idx_varreg, 15: -15, 15:-15], pxsizex*1e-3, fig = fig_bestres, ax=ax_bestres[2])
    ax_bestres[2].set_title('Variance Regularization - Mean')
    gra.ShowImg(adam_mean[idx_adam, 15: -15, 15:-15], pxsizex*1e-3, fig = fig_bestres, ax=ax_bestres[3])
    ax_bestres[3].set_title('Adam Optimization - Mean')
    gra.ShowImg(ground_truth[15: -15, 15:-15], pxsizex*1e-3, fig = fig_bestres, ax=ax_bestres[4])
    ax_bestres[4].set_title('Ground Truth')
    params_title = params.copy()
    params_title["num_iter"] = 'best'
    fig_bestres.suptitle(str(params_title)[1:-1], fontsize=16)
    
    # Compute PSNR
    psnr_rl = utils.peak_signal_to_noise_ratio(ground_truth, rl)
    psnr_var = utils.peak_signal_to_noise_ratio(ground_truth, varreg_mean)
        
    fig_psnr, ax_psnr = plt.subplots(1, 1, figsize=(20, 10))

    ax_psnr.plot(psnr_rl, color='r', label = 'rl')
    ax_psnr.plot(psnr_var, color='C2', label = 'varreg')
    
    psnr_adam = utils.peak_signal_to_noise_ratio(ground_truth, adam_mean)
    ax_psnr.plot(psnr_adam, color='C0', label = 'adam')

    ax_psnr.set_xlabel('Iteration')
    ax_psnr.set_ylabel('Normalized PSNR')
    ax_psnr.legend()
    ax_psnr.set_title(str(params)[1:-1])
    
    
    # Save the results
    os.makedirs(output_dir, exist_ok=True)

    rs.save_experiment_results(fig_res, fig_loss, fig_kl, fig_bestres, fig_psnr, params, output_dir, print_csv=print_csv)

    plt.close('all')
    return kl_var, kl_adam
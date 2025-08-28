import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import brighteyes_ism.analysis.Tools_lib as tools
import brighteyes_ism.analysis.Graph_lib as gra
from IPython.display import display, HTML
from matplotlib.patches import Circle
from PIL import Image

from . import varreg
from . import utils

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
        "W": params["W"],
        "method": params["method"],
        "initialization": params["initialization"],
        "Figure over K": f"<a href='{fig_paths[0]}' target='_blank'>Figure over K</a>",
        "Figure results": f"<a href='{fig_paths[1]}' target='_blank'>Figure results</a>",
        "Figure KL": f"<a href='{fig_paths[2]}' target='_blank'>Figure KL</a>",
        "Figure best iter": f"<a href='{fig_paths[3]}' target='_blank'>Figure best iter</a>",
    }

def save_experiment_results(fig_K, fig_results, fig_KL, fig_best, params, output_dir, plot_K=True, print_csv=True):

    """
    Save the results of an experiment to a file.
    
    Parameters:
        fig_K: Figure with all the K results.
        fig_results: Figure for the results.
        fig_KL: Figure for the KL divergence over iterations.
        fig_best: Figure for the best iteration result.
        tensor_PowSpec: Tensor for the power spectrum with dimensions (num_iter, kx, ky).
        tensor_noise_variance: Tensor for the noise variance with dimensions (num_iter, kx, ky).
        indices: Indices of iterations to save in the GIFs.
        params: Parameters of the experiment.
        output_dir: Output directory for saving the results.
        plot_K: Whether to plot the K figure.
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
    if plot_K:
        fig1_path = os.path.join(output_dir, f"{next_id}_figK.png")
    else:
        fig1_path = None
    fig2_path = os.path.join(output_dir, f"{next_id}_figResults.png")
    fig3_path = os.path.join(output_dir, f"{next_id}_figKL.png")
    fig4_path = os.path.join(output_dir, f"{next_id}_figBestIter.png")


    # Save figures
    if plot_K:
        fig_K.savefig(fig1_path)
    fig_results.savefig(fig2_path)
    fig_KL.savefig(fig3_path)
    fig_best.savefig(fig4_path)

    # New row
    new_entry = create_entry_dataframe(next_id, params, [fig1_path, fig2_path, fig3_path, fig4_path])
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
    varreg_solution = varreg.variance_reconstruction(dset, psf, lam=params["lambda"], W=weight,
                        max_iter=params["num_iter"], tol=None, rec_to_save='all', process='gpu', method=params["method"],
                        regularizer = params["regularizer"], normalize = False)
    varreg_mean = varreg_solution.mean(axis=-1)
    if compute_adam:
        adam_solution = varreg.adam_optimize_obj(dset, psf, num_iter=params["num_iter"], lambda_value=params["lambda"], alpha_value=1,
                        weight=params["W"], initialization = params["initialization"], regularizer = params["regularizer"], lr=params["lr"],
                        process='gpu', full_KL= False, rec_to_save='all', verbose=False)
        
        adam_optimize_obj(dset, psf, num_iter=500, lambda_value=0.01, alpha_value = 0.01, weight:str = 'id',
                      initialization: str = 'flat', regularizer: str ='l2', lr=1e-3,  process: str = 'gpu',
                      full_KL: bool =False, rec_to_save: str = 'last', verbose: bool =True)

    # Plots to save
    rec_idx=-1
    if plot_K:
        fig_K = gra.ShowDataset(varreg_solution[rec_idx], figsize = (8, 10))
        fig_K.suptitle(f"K={params['K']}, n_iter={params['num_iter'] if rec_idx==-1 else rec_idx}, lambda={params['lambda']}, W={params['W']}, method={params['method']}", fontsize=16)
    else:
        fig_K = None
        
    pxsizex = params["pixelsize"]
        
    fig_res, ax_res = plt.subplots(1, 4, figsize=(30, 7))
    gra.ShowImg(img, pxsizex*1e-3, fig = fig_res, ax=ax_res[0])
    ax_res[0].set_title('Raw Image')
    gra.ShowImg(rl[rec_idx], pxsizex*1e-3, fig = fig_res, ax=ax_res[1])
    ax_res[1].set_title('Richardson-Lucy')
    gra.ShowImg(varreg_mean[rec_idx], pxsizex*1e-3, fig = fig_res, ax=ax_res[2])
    ax_res[2].set_title('Variance Regularization - Mean')
    gra.ShowImg(ground_truth, pxsizex*1e-3, fig = fig_res, ax=ax_res[3])
    ax_res[3].set_title('Ground Truth')
    fig_res.suptitle(f"K={params['K']}, n_iter={params['num_iter'] if rec_idx==-1 else rec_idx}, \
                lambda={params['lambda']}, W={params['W']}, method={params['method']}", fontsize=16)
    
    # KL divergence
    kl_var = tools.kl_divergence(ground_truth, varreg_mean.numpy(), normalize_entries = True)
    
    #plot KL divergence
    fig_kl, ax_kl = plt.subplots(1, 1, figsize=(20, 10))

    idx_rl = np.argmin(kl_rl)
    idx_varreg = np.argmin(kl_var)

    ax_kl.plot(kl_rl, label = f'Richardson-Lucy KL_min={kl_rl[idx_rl]:.3f}')
    ax_kl.plot(kl_var, label = f'Variance Regularization - Mean KL_min={kl_var[idx_varreg]:.3f}')
    ax_kl.scatter(idx_rl, kl_rl[idx_rl])
    ax_kl.scatter(idx_varreg, kl_var[idx_varreg])
    ax_kl.set_xticks(list(ax_kl.get_xticks()[1:]) + [idx_rl])
    ax_kl.set_xticks(list(ax_kl.get_xticks()[1:]) + [idx_varreg])
    ax_kl.set_xlabel('Iteration')
    ax_kl.set_ylabel('Normalized KL')
    ax_kl.legend()
    ax_kl.set_title(f"K={params['K']}, n_iter={params['num_iter']}, lambda={params['lambda']}, \
                W={params['W']}, method={params['method']}", fontsize=16)
    
    # Plot best iteration
    fig_bestres, ax_bestres = plt.subplots(1, 4, figsize=(30, 7))
    gra.ShowImg(img, pxsizex*1e-3, fig = fig_bestres, ax=ax_bestres[0])
    ax_bestres[0].set_title('Raw Image')
    gra.ShowImg(rl[idx_rl], pxsizex*1e-3, fig = fig_bestres, ax=ax_bestres[1])
    ax_bestres[1].set_title('Richardson-Lucy')
    gra.ShowImg(varreg_mean[idx_varreg], pxsizex*1e-3, fig = fig_bestres, ax=ax_bestres[2])
    ax_bestres[2].set_title('Variance Regularization - Mean')
    gra.ShowImg(ground_truth, pxsizex*1e-3, fig = fig_bestres, ax=ax_bestres[3])
    ax_bestres[3].set_title('Ground Truth')
    fig_bestres.suptitle(f"K={params['K']}, n_iter='best', lambda={params['lambda']}, \
                    W={params['W']}, method={params['method']}", fontsize=16)
    
    # Save the results
    os.makedirs(output_dir, exist_ok=True)

    save_experiment_results(fig_K, fig_res, fig_kl, fig_bestres, params, output_dir, plot_K=plot_K, print_csv=print_csv)

    plt.close('all')
    return varreg_solution
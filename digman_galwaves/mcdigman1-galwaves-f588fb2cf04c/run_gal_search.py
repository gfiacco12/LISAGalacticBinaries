"""C 2023 Matthew C. Digman
code example to run the galactic binary parameter estimation pipeline
and plot results"""

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt

import likelihood_gb as trial_likelihood

from DTMCMC.ptmcmc_helpers import PTMCMCChain
from DTMCMC.temperature_ladder_helpers import TemperatureLadder
from DTMCMC.corr_summary_helpers import CorrelationSummary
from DTMCMC.proposal_strategy_helpers import ProposalStrategyParameters
from DTMCMC.proposal_manager_helper import get_default_proposal_manager
import ra_waveform_time as rwt
from instrument_noise import get_galactic_novar_noise_model
import wdm_const as wc 
from graph_helper import *
from ra_waveform_time import BinaryTimeWaveformAmpFreqD



if __name__ == '__main__':
    t0 = perf_counter()

    # starting variables
    n_chain = 2                 # number of total chains for parallel tempering
    n_cold = 2                         # number of T=1 chains for parallel tempering
    n_burnin =50000                   # number of iterations to discard as burn in
    block_size = 1000                  # number of iterations per block when advancing the chain state
    store_size = 200000                 # number of samples to store total
    N_blocks = store_size//block_size  # number of blocks the sampler must iterate through

    de_size = 5000                     # number of samples to store in the differential evolution buffer
    T_max = 20.                      # maximum temperature for geometric part of temperature ladder

    sigma_prior_lim = 100.              # minimum standard deviations to allow around prior in amplitude, frequency, and frequency derivative
    amp, alpha, beta, delta = rwt.TruthParamsCalculator(20.e-3, 0.8*wc.MSOLAR, 0.6*wc.MSOLAR, (2*wc.KPCSEC), (4.0*wc.SECSYEAR)) #not log of DL

    params_true = np.array([amp, alpha, beta, delta, np.log(2*wc.KPCSEC), 1.4*wc.MSOLAR, 0.6018927820922144*wc.MSOLAR, -0.26,  4.6, 0.25,  1.5,  1.6])  # true parameters for search -- Add in total mass and chirp mass
    print(alpha, beta, delta)
    # note that too many chains starting from the fiducial parameters can make the chain converge slower, if it fails to find secondary modes
    n_true_start = 0             # how many chains to start at params_true (0 for a blind search; the rest will start from prior draws)

    # create needed objects

    noise_AET_dense = get_galactic_novar_noise_model()                 # get the noise model for the galactic background best fit with no time variability
    strategy_params = ProposalStrategyParameters(de_size=de_size)      # get the object which stores various parameters related to the overall search strategy
    T_ladder = TemperatureLadder(n_chain, n_cold=n_cold, T_max=T_max)  # get the temperature ladder object

    like_obj = trial_likelihood.get_noiseless_gb_likelihood(params_true.copy(), noise_AET_dense, sigma_prior_lim, strategy_params)  # get the likelihood object
    params_true = like_obj.correct_bounds(params_true)                 # make sure the conventions on the parameters match
    print(like_obj.sigmas_in)
    logL_truths = like_obj.get_loglike(params_true)
    print("Log Likelihoods of truth parameters:", logL_truths)

    # create the starting samples
    starting_samples = np.zeros((T_ladder.n_chain, like_obj.n_par))
    for itrt in range(0, n_chain):
        if itrt < n_true_start:
            # use the true value
            starting_samples[itrt] = params_true
            print(like_obj.get_loglike(starting_samples[itrt]))
        else:
            # use a prior draw
            num_attempts = 0
            while True:
                starting_samples[itrt] = like_obj.prior_draw()
                current_sample = starting_samples[itrt]
                freq0 = alpha / (4.*wc.SECSYEAR)
                is_physical = betadelta_m1m2_check(current_sample[2], current_sample[3], freq0, (4.*wc.SECSYEAR), params_true[6], params_true[5])
                if is_physical == True:
                    break
                num_attempts += 1
                assert num_attempts < 500

    # create the overarching proposal manager object
    proposal_manager = get_default_proposal_manager(T_ladder, like_obj, strategy_params, starting_samples)

    print('Chain parameters', n_cold, n_chain, n_burnin, block_size, store_size, de_size, T_max)
    print('True Template SNR: ', like_obj.get_snr(params_true))

    # create the chain object
    mcc = PTMCMCChain(T_ladder, like_obj, strategy_params, block_size, store_size, starting_samples=starting_samples, params_true=params_true)

    t_init_end = perf_counter()
    print('all objects initialized in ', t_init_end-t0, 's')

    t_advance_begin = perf_counter()

    # the main loop which actually advances the MCMC state
    mcc.advance_N_blocks(N_blocks)

    t_advance_end = perf_counter()
    print('advanced in ', t_advance_end-t_advance_begin, 's')

    # generate some summary information
    corr_sum = CorrelationSummary()
    corr_sum.summarize_blocks(mcc, n_burnin)
    corr_sum.final_prints(mcc, n_burnin)
    #print("sigma scales:", mcc.proposal_manager.managers[0].sigma_scales)
    # get flattened samples for plotting
    samples_flattened, logLs_flattened, logLs_unflattened = mcc.get_stored_flattened(corr_sum.restrict_n_burnin(mcc, n_burnin)) 
            
    #makeHistogramofLogLike(logLs_flattened)

    #makeScatterPlot(logLs_flattened, samples_flattened[:,6])
    #iteration_number = np.linspace(0, store_size, len(logLs_flattened))
    # print(mcc.logL_means)
    plotChains(mcc.logL_means)

    #plt.semilogx(T_ladder.Ts,mcc.logL_vars[-1]*T_ladder.betas**2)

    tf = perf_counter()

    print('full search time ', str(tf-t0)+'s')

do_corner_plot = True
if do_corner_plot:
    # generate a corner plot
    import matplotlib.pyplot as plt
    import corner
    # reformat the samples to make the plots look nicer
    samples_format, params_true_format, labels = trial_likelihood.format_samples_output(samples_flattened, params_true, [rwt.idx_amp, rwt.idx_alpha, rwt.idx_beta, rwt.idx_delta])
    np.savetxt('recovery run.txt', samples_format)
    # create the corner plot figure
    fig = plt.figure(figsize=(10, 7.5))
    figure = corner.corner(samples_format, fig=fig, bins=25, hist_kwargs={"density": True}, show_titles=True, title_fmt=None,
                           title_kwargs={"fontsize": 12}, labels=labels, max_n_ticks=3, label_kwargs={"fontsize": 12}, labelpad=0.15,
                           smooth=0.25, levels=[0.682, 0.954], truths=params_true_format)
    # overplot the true parameters
    corner.overplot_points(figure, params_true_format[None], marker="s", color='tab:blue', markersize=4)
    corner.overplot_lines(figure, params_true_format, color='tab:blue')
    # adjust the figure to fit the box better
    fig.subplots_adjust(wspace=0., hspace=0., left=0.05, top=0.95, right=0.99, bottom=0.05)
    for ax in figure.get_axes():
        ax.tick_params(which='both', direction='in', bottom=True, top=True, left=True, right=True, labelsize=6)
    plt.show('prior_test.png')
    plt.savefig

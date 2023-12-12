set wildignore+=*/cwgf.egg-info/*,*.tar,*.tar.bkup,*/exps/*,*/wandb/*

function! RunTest(exp_name, run_py, gpu_id, args)
    let l:cwd = getcwd()
    let l:cmd = 'cd ' . l:cwd . '/tests/' . a:exp_name . ';'
    if (executable('conda') && !empty($CONDA_DEFAULT_ENV))
        let l:cmd = l:cmd . 'conda activate ' . $CONDA_DEFAULT_ENV . ';'
    endif
    let l:cmd = l:cmd . 'CUDA_VISIBLE_DEVICES=' . a:gpu_id  
    let l:cmd = l:cmd . ' python ' . a:run_py . ' '. a:args
    call VimuxRunCommand(l:cmd)
endfunction

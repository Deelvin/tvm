echo Downloading of DNNL packages (from Conda)

url_dnnl_v252_omp="https://anaconda.org/conda-forge/onednn/2.5.2/download/linux-64/onednn-2.5.2-omp_hf4ef041_0.tar.bz2"
url_dnnl_v252_tbb="https://anaconda.org/conda-forge/onednn/2.5.2/download/linux-64/onednn-2.5.2-tbb_h749a9ee_0.tar.bz2"


url_dnnl_v242_omp="https://anaconda.org/conda-forge/onednn/2.4.4/download/linux-64/onednn-2.4.4-omp_hf4ef041_0.tar.bz2"
url_dnnl_v242_tbb="https://anaconda.org/conda-forge/onednn/2.4.4/download/linux-64/onednn-2.4.4-tbb_h749a9ee_0.tar.bz2"

url_dnnl_v224_omp="https://anaconda.org/conda-forge/onednn/2.2.4/download/linux-64/onednn-2.2.4-omp_hf4ef041_0.tar.bz2"
url_dnnl_v224_tbb="https://anaconda.org/conda-forge/onednn/2.2.4/download/linux-64/onednn-2.2.4-tbb_h749a9ee_0.tar.bz2"

if ! command -v wget &> /dev/null
then
    echo "[ERROR] wget could not be found. Please install it."
    exit 1
fi

function download {
    dir=$1
    url=$2

    echo ${url}
    echo ${dir}

    mkdir __deps/${dir} -p 
    wget ${url} -P __deps/${dir}
    tar -xvf __deps/${dir}/*tar.bz2 -C __deps/${dir}
    rm __deps/${dir}/*tar.bz2
}

download dnnl_v2.5.2_omp ${url_dnnl_v252_omp}
download dnnl_v2.5.2_tbb ${url_dnnl_v252_tbb}

download dnnl_v2.4.2_omp ${url_dnnl_v242_omp}
download dnnl_v2.4.2_tbb ${url_dnnl_v242_tbb}

download dnnl_v2.2.4_omp ${url_dnnl_v224_omp}
download dnnl_v2.2.4_tbb ${url_dnnl_v224_tbb}
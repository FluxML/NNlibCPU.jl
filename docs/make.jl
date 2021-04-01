using NNlibCPU
using Documenter

DocMeta.setdocmeta!(NNlibCPU, :DocTestSetup, :(using NNlibCPU); recursive=true)

makedocs(;
    modules=[NNlibCPU],
    authors="chriselrod <elrodc@gmail.com> and contributors",
    repo="https://github.com/chriselrod/NNlibCPU.jl/blob/{commit}{path}#{line}",
    sitename="NNlibCPU.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://chriselrod.github.io/NNlibCPU.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/chriselrod/NNlibCPU.jl",
)

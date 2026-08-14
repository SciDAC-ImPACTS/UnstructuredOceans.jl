using Documenter
using UnstructuredOceans

DocMeta.setdocmeta!(UnstructuredOceans, :DocTestSetup, :(using UnstructuredOceans); recursive=true)

makedocs(;
    modules  = [UnstructuredOceans],
    authors  = "SciDAC ImPACTS Team",
    sitename = "UnstructuredOceans.jl",
    format   = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        edit_link  = "main",
        assets     = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Tutorials" => [
            "Getting started" => "tutorials/getting_started.md",
        ],
        "How-to guides" => [
            "Choose a compute backend" => "howto/choose_backend.md",
            "Automatic differentiation" => "howto/automatic_differentiation.md",
            "Configure a simulation"    => "howto/configuration.md",
            "Run the benchmarks"        => "howto/run_benchmarks.md",
        ],
        "Explanation" => [
            "Governing equations"   => "explanation/governing_equations.md",
            "TRiSK discretization"  => "explanation/trisk_discretization.md",
            "Software architecture" => "explanation/architecture.md",
        ],
        "Reference" => [
            "API" => "reference/api.md",
        ],
    ],
    # Verify every *exported* symbol is documented (internal helpers are exempt).
    checkdocs = :exports,
)

deploydocs(;
    repo      = "github.com/SciDAC-ImPACTS/Moka.jl",
    devbranch = "main",
    push_preview = true,
)

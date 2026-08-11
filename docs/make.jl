using Documenter
using MOKA

DocMeta.setdocmeta!(MOKA, :DocTestSetup, :(using MOKA); recursive=true)

makedocs(;
    modules  = [MOKA],
    authors  = "SciDAC ImPACTS Team",
    sitename = "MOKA.jl",
    format   = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical  = "https://GodotMisogi.github.io/Moka.jl",
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
    repo      = "github.com/SciDAC/Moka.jl",
    devbranch = "main",
    push_preview = true,
)

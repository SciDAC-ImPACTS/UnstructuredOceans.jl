# %% Dependencies
# Reuse the Munk exact solution, numerical streamfunction reconstruction, and the
# area-weighted L2 error from analysis.jl (included as a library — its main block
# is guarded, so nothing runs on include).
include("analysis.jl")

using DataFrames

# Spatial convergence of the barotropic (Munk) gyre. The coarsest mesh is
# included for context; only resolutions below the ~58 km boundary-layer width
# should be used to interpret the asymptotic order.
const RESOLUTIONS = ["5km", "10km", "20km", "40km"]

function convergence_data(resolutions)
    rows = map(resolutions) do resolution
        directory = joinpath(@__DIR__, resolution)
        results, mesh = read_and_compare(joinpath(directory, "output.nc"),
                                         joinpath(directory, "initial_state.nc"))
        spacing_km = mean(Array(mesh["dcEdge"][:])) / 1e3
        close(mesh)
        return (; dc = spacing_km, l2_err = results["l2_err"])
    end
    data = DataFrame(rows)
    sort!(data, :dc)
    return data
end

function convergence_fit(data)
    coefficients = hcat(log10.(data.dc), ones(nrow(data))) \ log10.(data.l2_err)
    return (; order = coefficients[1], values = 10^coefficients[2] .* data.dc .^ coefficients[1])
end

function convergence_plot(data, fit)
    reference = 0.5 .* data.l2_err[end] .* (data.dc ./ data.dc[end])
    fig = Figure(size = (500, 400))
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10,
              title = "Barotropic Gyre (Munk, $(BC))",
              xlabel = "Resolution (km)", ylabel = "Streamfunction L2 error")

    lines!(ax, data.dc, reference; color = :black, linestyle = :dash,
           alpha = 0.3, label = "First order")
    lines!(ax, data.dc, reference .* (data.dc ./ data.dc[end]); color = :black,
           alpha = 0.3, label = "Second order")
    lines!(ax, data.dc, fit.values; color = :black,
           label = "Linear fit (order = $(round(fit.order; digits = 3)))")
    scatter!(ax, data.dc, data.l2_err; marker = :circle, label = "L2 error")
    axislegend(ax; position = :lt)
    return fig
end

data = convergence_data(RESOLUTIONS)
fit = convergence_fit(data)
fig = convergence_plot(data, fit)
save(joinpath(@__DIR__, "convergence.png"), fig; px_per_unit = 3)
display(fig)
println("Barotropic gyre streamfunction convergence order: $(round(fit.order; digits = 3))")

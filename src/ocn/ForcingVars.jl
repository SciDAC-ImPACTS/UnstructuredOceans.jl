import MOKA.MPASMesh: padded_index_array

const Abstract1DArray = AbstractArray{F, 1} where F <: Real
const Array1DOrNothing = Union{Abstract1DArray, Nothing}

struct ForcingVars{OA}

    # var: Total surface stress on the ocean defined at edge midpoints and
    #      pointing in the direction of the edge normal. Field is the sum of
    #      constituent stresses (e.g. wind stress) and is used to compute the
    #      tendency in the normal velocity
    # dim: nEdges [time eventually 
    # units: N m$^{-2}$
    surfaceStress::OA
end

# null constructor
ForcingVars() = ForcingVars(nothing)

function ForcingVars(config::GlobalConfig, mesh::Mesh; backend=KA.CPU())
    
    # try reading the forcing section of the streams
    local forcingConfig
    try 
        forcingConfig = ConfigGet(config.streams, "forcing")
    catch e
        # if forcing section is missing return null forcing object
        if isa(e, KeyError)
            return ForcingVars()
        # throw error if something other a KeyError (i.e. missing section)
        else
            rethrow()
        end
    end

    forcing_filename = ConfigGet(forcingConfig, "filename_template")
    
    # Can forcing "type" be anything other than "input"?
    # TODO: Support time dependent forcing
    if ConfigGet(forcingConfig, "input_interval") != "initial_only"
        @error "Time dependent forcing is, curently, NOT supported."
    end

    # calculate the bulk forcing form the forcing file
    sfcStress = surface_bulk_forcing_vel(forcing_filename, mesh)

    Adapt.adapt(backend, ForcingVars(sfcStress))
end

# For empty forcing just return the empty forcing structr
Adapt.adapt_structure(to, x::ForcingVars{OA}) where OA <: Nothing = x

Adapt.adapt_structure(to, x::ForcingVars) = ForcingVars(Adapt.adapt(to, x.surfaceStress))

function surface_bulk_forcing_vel(forcing_fn::String, mesh::Mesh)
   # unpack the mesh arrays needed to compute bulk forcing
   @unpack nCells = mesh.HorzMesh.PrimaryCells
   @unpack nEdges, angleEdge, cellsOnEdge = mesh.HorzMesh.Edges

   # open the forcing file
   forcing_ds = NCDataset(forcing_fn)
    
   f(var) = haskey(forcing_ds, var)
   if !all(map(v -> haskey(forcing_ds, v), ["windStressZonal", "windStressMeridional"]))
       @error "Forcing dataset does not contain the variables needed"
   end

   # Pre-allocate zero indexed offsetarrays on the CPU 
   windStressZonal = padded_index_array(nCells; eltype=Float64)
   windStressMerid = padded_index_array(nCells; eltype=Float64)

   # read the wind stress components from the forcing file
   windStressZonal[1:end] = forcing_ds["windStressZonal"][:, 1]
   windStressMerid[1:end] = forcing_ds["windStressMeridional"][:, 1]

   # allocate the surface stress array to store the bulk forcing in 
   # TODO: paramertize the type to support single precision runs
   sfcStress = zeros(Float64, nEdges)

   for iEdge in 1:nEdges
       iCell1 = cellsOnEdge[1, iEdge]
       iCell2 = cellsOnEdge[2, iEdge]

       zonalWSEdge = 0.5 * (windStressZonal[iCell1] + windStressZonal[iCell2])
       meridWSEdge = 0.5 * (windStressMerid[iCell1] + windStressMerid[iCell2])
       
       sfcStress[iEdge] = cos(angleEdge[iEdge]) * zonalWSEdge +
                          sin(angleEdge[iEdge]) * meridWSEdge
   end

   # return the surface stress (i.e. bulk forcing)
   return sfcStress
end

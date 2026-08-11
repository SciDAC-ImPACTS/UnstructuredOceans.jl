# define our parent abstract type 
abstract type PressureGradient end

using KernelAbstractions
const KA=KernelAbstractions

# define the supported PressureGradient types to dispatch on. 
abstract type sshGradient <: PressureGradient end 

function pressure_gradient_tendency!(Tend::TendencyVars,
                                     Prog::PrognosticVars,
                                     Diag::DiagnosticVars,
                                     Mesh::Mesh,
                                     ::Type{sshGradient};
                                     nthreads=DEFAULT_NTHREADS)
    backend = KA.get_backend(Tend.tendNormalVelocity)

    @unpack HorzMesh, VertMesh = Mesh
    @unpack PrimaryCells, DualCells, Edges = HorzMesh

    @unpack maxLevelEdge = VertMesh
    @unpack nEdges, dcEdge, cellsOnEdge, boundaryEdge = Edges

    # gravity as a 1-element (inactive) device array, like momentumDel2 — a bare
    # Float64 kernel arg is classified Active by Enzyme reverse mode.
    gravity = Mesh.Constants.gravity

    ssh = Prog.ssh[end]
    @unpack tendNormalVelocity = Tend

    kernel! = SSHGradOnEdge!(backend, nthreads)
    kernel!(tendNormalVelocity,
            ssh,
            cellsOnEdge,
            dcEdge,
            boundaryEdge,
            maxLevelEdge.Top,
            gravity,
            ndrange=nEdges)

    @pack! Tend = tendNormalVelocity
end

@kernel function SSHGradOnEdge!(tendency,
                                ssh,
                                cellsOnEdge,
                                dcEdge,
                                boundaryEdge,
                                maxLevelEdgeTop,
                                gravity)

    iEdge = @index(Global, Linear)

    if boundaryEdge[iEdge] != 1
        @inbounds jCell1 = cellsOnEdge[1,iEdge]
        @inbounds jCell2 = cellsOnEdge[2,iEdge]

        @inbounds InvDcEdge = 1.0 / dcEdge[iEdge]

        @inbounds g = gravity[1]
        for k in 1:maxLevelEdgeTop[iEdge]
            tendency[k, iEdge] -= g * InvDcEdge * (ssh[jCell2] - ssh[jCell1])
        end
    end
end

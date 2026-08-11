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

    ssh = Prog.ssh[end]
    @unpack tendNormalVelocity = Tend

    kernel! = SSHGradOnEdge!(backend, nthreads)
    kernel!(tendNormalVelocity,
            ssh,
            cellsOnEdge,
            dcEdge,
            boundaryEdge,
            maxLevelEdge.Top,
            ndrange=nEdges)

    @pack! Tend = tendNormalVelocity
end

@kernel function SSHGradOnEdge!(tendency,
                                ssh,
                                cellsOnEdge,
                                dcEdge,
                                boundaryEdge,
                                maxLevelEdgeTop)

    iEdge = @index(Global, Linear)

    if boundaryEdge[iEdge] != 1
        @inbounds jCell1 = cellsOnEdge[1,iEdge]
        @inbounds jCell2 = cellsOnEdge[2,iEdge]

        @inbounds InvDcEdge = 1.0 / dcEdge[iEdge]

        for k in 1:maxLevelEdgeTop[iEdge]
            tendency[k, iEdge] -= 9.80616 * InvDcEdge * (ssh[jCell2] - ssh[jCell1])
        end
    end
end

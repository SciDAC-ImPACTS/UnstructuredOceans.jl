# define our parent abstract type 
abstract type PresssureGradient end

using KernelAbstractions
const KA=KernelAbstractions

# define the supported PressureGradient types to dispatch on. 
abstract type sshGradient <: PresssureGradient end 

function pressure_gradient_tendency!(Tend::TendencyVars,
                                     Prog::PrognosticVars,
                                     Diag::DiagnosticVars,
                                     Mesh::Mesh,
                                     ::Type{sshGradient};
                                     backend = KA.CPU())

    @unpack HorzMesh, VertMesh = Mesh
    @unpack PrimaryCells, DualCells, Edges = HorzMesh

    @unpack maxLevelEdge = VertMesh
    @unpack nEdges, dcEdge, cellsOnEdge, boundaryEdge = Edges

    ssh = Prog.ssh[end]
    @unpack tendNormalVelocity = Tend

    nthreads = 50
    kernel! = SSHGradOnEdge!(backend, nthreads)
    kernel!(tendNormalVelocity,
            ssh,
            cellsOnEdge,
            dcEdge,
            boundaryEdge,
            maxLevelEdge.Top,
            ndrange=nEdges)
    KA.synchronize(backend)

    @pack! Tend = tendNormalVelocity
end

@kernel function SSHGradOnEdge!(tendency,
                                ssh,
                                cellsOnEdge,
                                dcEdge,
                                boundaryEdge,
                                maxLevelEdgeTop)

    iEdge = @index(Global, Linear)

    if boundaryEdge[iEdge] == 1
        return
    end

    @inbounds jCell1 = cellsOnEdge[1,iEdge]
    @inbounds jCell2 = cellsOnEdge[2,iEdge]

    @inbounds InvDcEdge = 1.0 / dcEdge[iEdge]

    for k in 1:maxLevelEdgeTop[iEdge]
        tendency[k, iEdge] -= 9.80616 * InvDcEdge * (ssh[jCell2] - ssh[jCell1])
    end
end

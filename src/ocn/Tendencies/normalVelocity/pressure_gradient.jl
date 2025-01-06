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
    @unpack nEdges, dcEdge, cellsOnEdge, edgeMask = Edges
   
    # get the current timelevel of ssh 
    ssh = Prog.ssh[end] #[:,end]
    # unpack the normal velocity tendency term
    @unpack tendNormalVelocity = Tend 
    
    # initialize the kernel
    nthreads = 50
    kernel! = SSHGradOnEdge!(backend, nthreads)
    # use kernel to compute gradient
    kernel!(tendNormalVelocity,
            ssh,
            cellsOnEdge,
            dcEdge,
            maxLevelEdge.Top,
            edgeMask,
            ndrange=nEdges)
    # sync the backend 
    KA.synchronize(backend)
    
    # pack the tendecy pack into the struct for further computation
    @pack! Tend = tendNormalVelocity 
end

@kernel function SSHGradOnEdge!(tendency,
                                @Const(ssh),
                                @Const(cellsOnEdge), 
                                @Const(dcEdge), 
                                @Const(maxLevelEdgeTop),
                                @Const(edgeMask))

    # global indices over nEdges
    iEdge = @index(Global, Linear)

    # cell connectivity information for iEdge
    @inbounds jCell1 = cellsOnEdge[1,iEdge]      
    @inbounds jCell2 = cellsOnEdge[2,iEdge]
    
    # inverse edge spacing for iEdge
    @inbounds InvDcEdge = 1. / dcEdge[iEdge]
  
    for k in 1:maxLevelEdgeTop[iEdge]
        # gradient on edges calculation 
        tendency[k, iEdge] -= 9.80616 * edgeMask[k, iEdge] * \
                              (ssh[jCell2] - ssh[jCell1]) * InvDcEdge
    end
end

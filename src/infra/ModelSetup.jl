using UnstructuredOceans.MPASMesh

# structure for storing Model/Simulation structures 
struct ModelSetup    
    config::GlobalConfig
    mesh::Mesh
    timeManager::Clock
    #constants::Constants
end 

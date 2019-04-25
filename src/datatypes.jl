#=
A key for remembering the dimensions of each data type for a grid with
NX, NY interior cells:

CellData: (NX, NY) interior, (NX+2,NY+2) total
          interior indexing 2:NX+1,2:NY+1 (and no boundary)
EdgeData: x component: (NX+1,NY) interior/boundary, (NX+1,NY+2) total
                       so interior indexing 2:NX,2:NY+1
                       and boundary at [1,2:NY+1], [NX+1,2:NY+1]
          y component: (NX,NY+1) interior/boundary, (NX+2,NY+1) total
                       so interior indexing 2:NX+1,2:NY
                       and boundary at [2:NX+1,1], [2:NX+1,NY+1]
NodeData: (NX+1,NY+1) interior/boundary/corner and total (no ghosts)
          interior indexing 2:NX,2:NY
          and boundaries at [1,2:NY], [NX+1,2:NY], [2:NX,1], [2:NX,NY+1]
          and corners at [1,1], [NX+1,1], [1,NY+1], [NX+1,NY+1]

Also, the index spaces of each of these variables are a little different from
each other. Treating Ic as a continuous variable in the cell center index space, and
In as a continuous variable in node index space, then our convention here is that

        i_c = i_n + 1/2

In other words, a node with index i_n = 1 corresponds to a location in the cell
center space at i_c = 3/2.
=#

# We import the + and - operations from Julia so that we can extend them to
# our new data types here
import Base:+, -,*,/

export CellData, NodeData, EdgeData, XEdgeData, YEdgeData, ScalarData, VectorData, GridData

#================= THE DATA TYPES =================#

#=
We will be a bit more sophisticated with our typing system now. The data
are organized as follows:
                               |-> CellData
            |->  ScalarData -> |-> NodeData
GridData -> |                  |-> XEdgeData
            |                  |-> YEdgeData
            |
            |-> VectorData     |-> EdgeData (with fields XEdgeData, YEdgeData)

The ScalarData all have a single field `data`.
=#

# Here we defined ScalarData. Note that we have declared
# it to be a subtype of AbstractMatrix. This will allow us to do array-like
# things on scalar-valued grid data.
abstract type ScalarData{NX,NY} <: AbstractMatrix{Float64} end

# We also create VectorData as a parent type. This is also
# a subtype of AbstractMatrix.
abstract type VectorData{NX,NY} <: AbstractMatrix{Float64} end

#=
Now, we will create GridData to be a union of scalar and vector types. This
makes it easier to dispatch for cases in which it does not matter what specific
type of data we are passing in.
=#
GridData = Union{ScalarData,VectorData}


#= Here we are constructing a data type for data that live at the cell
centers on a grid.
=#
struct CellData{NX,NY} <: ScalarData{NX,NY}
  data :: Array{Float64,2}
end


#=
The following are new types in our hierarchy. They are the components of edge data. But
since we want them to behave like matrices/vectors sometimes, we declare them
to be subtypes of ScalarData.
=#
struct XEdgeData{NX,NY} <: ScalarData{NX,NY}
  data :: Array{Float64,2}
end

struct YEdgeData{NX,NY} <: ScalarData{NX,NY}
  data :: Array{Float64,2}
end

#=
Here we are constructing a data type for cell edges. Notice that we use
NX and NY again as parameters. These still correspond to the number of interior
cells, so that edge data associated with other data types on the same grid
share the same parameter values. We have separate arrays for the x and y
components.
=#
struct EdgeData{NX,NY} <: VectorData{NX,NY}
  qx :: XEdgeData{NX,NY}
  qy :: YEdgeData{NX,NY}
end

#=
Node-based data. Again, NX and NY parameters correspond to number of
interior cells. Again, it is a subtype of ScalarData{NX,NY}.
=#
struct NodeData{NX,NY} <: ScalarData{NX,NY}
  data :: Array{Float64,2}
end



#================  CELL-CENTERED DATA CONSTRUCTORS ====================#

# This is called a constructor function for the data type. Here, it
# simply fills in the parameters NX and NY and returns the data in the new type.
"""
    CellData(data)

Set up a type of data that sit at cell centers. The `data` include the
interior cells and the ghost cells, so the resulting grid will be smaller
by 2 in each direction.

Example:
```
julia> w = ones(5,4);

julia> CellData(w)
5×4 CellData{3,2}:
 1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0
 ```
"""
function CellData(data::Array{Float64,2})
  nx_pad, ny_pad = size(data)
  NX, NY = nx_pad-2, ny_pad-2

  return CellData{NX,NY}(data)
end

#== Some other constructors for cell data ==#

# This constructor function allows us to just specify the interior grid size.
# It initializes with zeros
"""
    CellData(nx,ny)

Set up cell centered data equal to zero on a grid with `(nx,ny)` interior cells.
Pads with a layer of ghost cells on all sides.
"""
CellData(nx::Int,ny::Int) = CellData{nx,ny}(zeros(nx+2,ny+2))

# This constructor function is useful when we already have some grid data
# It initializes with zeros
"""
    CellData(p::GridData)

Set up cell centered data equal to zero on a grid corresponding to supplied
grid data `p`. Pads with a layer of ghost cells on all sides.
"""
CellData(p::ScalarData{NX,NY}) where {NX,NY} = CellData(NX,NY)
CellData(p::VectorData{NX,NY}) where {NX,NY} = CellData(NX,NY)



#================  EDGE DATA CONSTRUCTORS ====================#

# Create blank sets of x and y edge data of the appropriate size
XEdgeData(nx::Int,ny::Int) = XEdgeData{nx,ny}(zeros(nx+1,ny+2))
YEdgeData(nx::Int,ny::Int) = YEdgeData{nx,ny}(zeros(nx+2,ny+1))

# Provide the data directly
function XEdgeData(data::Array{Float64,2})
  nx_pad, ny_pad = size(data)
  NX, NY = nx_pad-1, ny_pad-2

  return XEdgeData{NX,NY}(data)
end
function YEdgeData(data::Array{Float64,2})
  nx_pad, ny_pad = size(data)
  NX, NY = nx_pad-2, ny_pad-1

  return YEdgeData{NX,NY}(data)
end

"""
    EdgeData(nx,ny)

Set up edge data equal to zero on a grid with `(nx,ny)` interior cells. Pads
with ghosts where appropriate.
"""
EdgeData(nx::Int,ny::Int) = EdgeData{nx,ny}(XEdgeData(nx,ny),YEdgeData(nx,ny))

"""
    EdgeData(p::GridData)

Set up edge data equal to zero on a grid of a size corresponding to the
given grid data `p`. Pads with ghosts.

Example:
```
julia> p = CellData(5,4)
7×6 CellData{5,4}:
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0

julia> q = EdgeData(p)
EdgeData{5,4}([0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0], [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0])

julia> q.qx
6×6 XEdgeData{5,4}:
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0

 julia> q.qy
 7×5 YEdgeData{5,4}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
```
"""
EdgeData(p::ScalarData{NX,NY}) where {NX,NY} = EdgeData(NX,NY)
EdgeData(p::VectorData{NX,NY}) where {NX,NY} = EdgeData(NX,NY)

# If the arrays themselves for qx and qy are provided:
function EdgeData(qx::Array{Float64,2},qy::Array{Float64,2})
  nx_qx, ny_qx = size(qx)
  nx_qy, ny_qy = size(qy)
  NX, NY = nx_qx-1,ny_qx-2
  # Need to make sure these qx and qy are compatible with each other:
  @assert NX == nx_qy-2 && NY == ny_qy-1
  return EdgeData{NX,NY}(XEdgeData(qx),YEdgeData(qy))
end

#================  NODE DATA CONSTRUCTORS ====================#

"""
    NodeData(nx,ny)

Set up node data equal to zero on a grid with `(nx,ny)` interior cells. Note
that node data has no ghosts.
"""
NodeData(nx::Int,ny::Int) = NodeData{nx,ny}(zeros(nx+1,ny+1))

"""
    NodeData(p::GridData)

Set up node data equal to zero on a grid of a size corresponding to the
given grid data `p`.

Example:
```
julia> p = CellData(5,4)
7×6 CellData{5,4}:
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0

julia> w = NodeData(p)
6×5 NodeData{5,4}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 ```
"""
NodeData(p::ScalarData{NX,NY}) where {NX,NY} = NodeData(NX,NY)
NodeData(p::VectorData{NX,NY}) where {NX,NY} = NodeData(NX,NY)


# if the data themselves are provided as an array
function NodeData(data::Array{Float64,2})
  nxnode, nynode = size(data)
  return NodeData{nxnode-1,nynode-1}(data)
end


#====== SOME BASIC THINGS WE SHOULD BE ABLE TO DO ON THIS DATA =======#

#=
We need to explicitly tell Julia how to do some things on our new data types
=#

Base.size(p::ScalarData) = size(p.data)
Base.length(p::ScalarData) = length(p.data)

Base.size(q::VectorData) = (length(q.qx)+length(q.qy),1)
Base.length(q::VectorData) = length(q.qx) + length(q.qy)

# The following four lines allow us to index grid data just as though it were an array.
# We can access the data with either one index (column-wise) or two
Base.getindex(p::ScalarData, i::Int) = p.data[i]
Base.setindex!(p::ScalarData, v, i::Int) = p.data[i] = convert(Float64, v)
Base.getindex(p::ScalarData, I::Vararg{Int, 2}) = p.data[I...]
Base.setindex!(p::ScalarData, v, I::Vararg{Int, 2}) = p.data[I...] = convert(Float64, v)

Base.getindex(p::VectorData, i::Int) =
    i > length(p.qx) ? p.qy[i-length(p.qx)] : p.qx[i]
Base.setindex!(p::VectorData, v, i::Int) =
    i > length(p.qx) ? p.qy[i-length(p.qx)] = convert(Float64, v) : p.qx[i] = convert(Float64, v)
Base.IndexStyle(::Type{<:VectorData}) = IndexLinear()

# Set it to negative of itself
function (-)(p_in::ScalarData)
  p = deepcopy(p_in)
  p.data .= -p.data
  return p
end

function (-)(p_in::EdgeData)
  p = deepcopy(p_in)
  p.qx .= -p.qx
  p.qy .= -p.qy
  return p
end

# Add and subtract the same type
function (-)(p1::T,p2::T) where {T<:ScalarData}
  return T(p1.data .- p2.data)
end

function (+)(p1::T,p2::T) where {T<:ScalarData}
  return T(p1.data .+ p2.data)
end

function (-)(p1::T,p2::T) where {T<:EdgeData}
  return T(p1.qx - p2.qx, p1.qy - p2.qy)
end

function (+)(p1::T,p2::T) where {T<:EdgeData}
  return T(p1.qx + p2.qx, p1.qy + p2.qy)
end

# Multiply and divide by a constant
function (*)(p::T,c::Number) where {T<:ScalarData}
  return T(c*p.data)
end


function (/)(p::T,c::Number) where {T<:ScalarData}
  return T(p.data ./ c)
end

function (*)(p::T,c::Number) where {T<:EdgeData}
  return T(c*p.qx,c*p.qy)
end

(*)(c::Number,p::T) where {T<:GridData} = *(p,c)

function (/)(p::T,c::Number) where {T<:EdgeData}
  return T(p.qx / c, p.qy / c)
end

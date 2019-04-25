export indices

"""
    indices(p::GridData,dir::Integer;[interior=true]) -> Range

Returns the range of indices in direction `dir` appropropriate for the type of
grid data in `p`. If only the interior range is needed (i.e. without boundary
or ghost values) set `interior=true`. For VectorData, the ranges for each component
are returned as a tuple.
"""
function indices(p::CellData{NX,NY},dir::Integer;interior::Bool=true) where {NX,NY}

  N = dir==1 ? NX : NY

  if interior
    return 2:(N+1)
  else
    return 1:(N+2)
  end

end

function indices(p::NodeData{NX,NY},dir::Integer;interior::Bool=true) where {NX,NY}

  N = dir==1 ? NX : NY

  if interior
    return 2:N
  else
    return 1:(N+1)
  end

end

function indices(p::EdgeData{NX,NY},dir::Integer;interior::Bool=true) where {NX,NY}

  # return x, y edge component ranges as a tuple

  if dir == 1
    if interior
      return 2:NX, 2:(NX+1)
    else
      return 1:(NX+1),1:(NX+2)
    end
  elseif dir == 2
    if interior
      return 2:(NY+1), 2:NY
    else
      return 1:(NY+2),1:(NY+1)
    end
  end
end

// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "hdmitsuba/curves.h"

#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/types.h>
#include <pxr/imaging/hd/basisCurves.h>
#include <pxr/imaging/hd/basisCurvesSchema.h>
#include <pxr/imaging/hd/basisCurvesTopology.h>
#include <pxr/imaging/hd/basisCurvesTopologySchema.h>
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/materialBindingSchema.h>
#include <pxr/imaging/hd/materialBindingsSchema.h>
#include <pxr/imaging/hd/primvarsSchema.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/sceneIndex.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/hd/types.h>
#include <pxr/imaging/hd/visibilitySchema.h>
#include <pxr/imaging/hd/xformSchema.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/path.h>

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "hdmitsuba/debug_codes.h"
#include "hdmitsuba/render_param.h"
#include "hdmitsuba/scene_manager.h"
#include "hdmitsuba/spec_types.h"
#include "hdmitsuba/utils.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

float CalculateMeanWidth(const VtFloatArray& widths,
                         float default_radius = 0.01f) {
  if (widths.empty()) return default_radius;
  double sum = 0.0;
  for (float w : widths) {
    sum += w;
  }
  return static_cast<float>(sum / static_cast<double>(widths.size()));
}

template <typename WidthFn>
std::vector<float> PackControlPoints(const VtVec3fArray& points,
                                     WidthFn&& width_fn) {
  std::vector<float> control_points(points.size() * 4);
  for (size_t i = 0; i < points.size(); ++i) {
    const GfVec3f& point = points[i];
    control_points[4 * i + 0] = point[0];
    control_points[4 * i + 1] = point[1];
    control_points[4 * i + 2] = point[2];
    control_points[4 * i + 3] = width_fn(i);
  }
  return control_points;
}

// Converts Catmull-Rom strands into disjoint 4-point B-spline segments.
// Segment s of a strand reads the Catmull-Rom window [P_s, ..., P_{s+3}] and
// maps it through M_bspline^-1 * M_catmullRom to the equivalent B-spline
// control points. With `wrap = pinned`, the windows also cover the phantom
// points P_{-1} = 2 P_0 - P_1 and P_N = 2 P_{N-1} - P_{N-2}, so the spline
// starts at P_0 and ends at P_{N-1} (N - 1 segments, valid from N = 2). The
// phantom radius copies the endpoint radius instead of extrapolating it, which
// would turn negative at tips that taper to zero.
//
// Adjacent B-spline segments cannot share control points after the change of
// basis, so each segment stores its own 4 points (4x the control point memory
// of a shared layout), and the resulting `control_points` no longer map 1:1 to
// the USD points. The result is returned as equivalent B-spline strands of 4
// points each, so the regular B-spline path builds the segment indices. The
// caller must ensure sum(vertex_counts) matches the number of control points.
//
// Keep in sync with _catmull_rom_to_bspline in usd_mitsuba/translator.py.
void ConvertCatmullRomToBSpline(bool pinned, VtIntArray& vertex_counts,
                                std::vector<float>& control_points) {
  static constexpr float kCatmullRomToBSpline[4][4] = {
      {7.0f / 6.0f, -4.0f / 6.0f, 5.0f / 6.0f, -2.0f / 6.0f},
      {-2.0f / 6.0f, 11.0f / 6.0f, -4.0f / 6.0f, 1.0f / 6.0f},
      {1.0f / 6.0f, -4.0f / 6.0f, 11.0f / 6.0f, -2.0f / 6.0f},
      {-2.0f / 6.0f, 5.0f / 6.0f, -4.0f / 6.0f, 7.0f / 6.0f},
  };
  const int first = pinned ? -1 : 0;
  size_t num_segments = 0;
  for (int count : vertex_counts.AsConst()) {
    num_segments += std::max(0, count - 3 - 2 * first);
  }
  std::vector<float> bspline_points(num_segments * 16);
  float* out = bspline_points.data();
  int offset = 0;
  for (int count : vertex_counts.AsConst()) {
    // Window indices run from `first` to `count - 1 - first`.
    for (int start = first; start + 3 <= count - 1 - first; ++start) {
      float window[4][4];
      for (int k = 0; k < 4; ++k) {
        const int local = start + k;
        const int clamped = std::clamp(local, 0, count - 1);
        // In-range indices have clamped == local, so 2*anchor - neighbor ==
        // anchor. Out-of-range indices (pinned only) reflect the endpoint
        // through its neighbor while preserving the endpoint radius.
        const float* anchor = &control_points[4 * (offset + clamped)];
        const float* neighbor =
            &control_points[4 * (offset + 2 * clamped - local)];
        for (int axis = 0; axis < 3; ++axis) {
          window[k][axis] = 2.0f * anchor[axis] - neighbor[axis];
        }
        window[k][3] = anchor[3];
      }
      for (int row = 0; row < 4; ++row, out += 4) {
        for (int col = 0; col < 4; ++col) {
          out[col] = kCatmullRomToBSpline[row][0] * window[0][col] +
                     kCatmullRomToBSpline[row][1] * window[1][col] +
                     kCatmullRomToBSpline[row][2] * window[2][col] +
                     kCatmullRomToBSpline[row][3] * window[3][col];
        }
      }
    }
    offset += count;
  }
  vertex_counts = VtIntArray(num_segments, 4);
  control_points = std::move(bspline_points);
}

}  // namespace

HdMitsubaCurves::HdMitsubaCurves(const SdfPath& id) : HdBasisCurves(id) {}

HdDirtyBits HdMitsubaCurves::GetInitialDirtyBitsMask() const {
  return HdChangeTracker::Clean | HdChangeTracker::DirtyPrimvar |
         HdChangeTracker::DirtyRepr | HdChangeTracker::DirtyTopology |
         HdChangeTracker::DirtyTransform | HdChangeTracker::DirtyVisibility |
         HdChangeTracker::DirtyMaterialId;
}

void HdMitsubaCurves::Sync(HdSceneDelegate* sceneDelegate,
                           HdRenderParam* renderParam, HdDirtyBits* dirtyBits,
                           const TfToken& /*reprToken*/) {
  static const HdDataSourceLocator points_locator(
      HdPrimvarsSchema::GetSchemaToken(), HdTokens->points,
      HdPrimvarSchemaTokens->primvarValue);
  static const HdDataSourceLocator widths_locator(
      HdPrimvarsSchema::GetSchemaToken(), HdTokens->widths,
      HdPrimvarSchemaTokens->primvarValue);
  static const HdDataSourceLocator material_locator(
      HdMaterialBindingsSchema::GetSchemaToken(),
      HdMaterialBindingsSchemaTokens->allPurpose,
      HdMaterialBindingSchemaTokens->path);
  static const HdDataSourceLocator basis_locator(
      TfToken("basisCurves"), TfToken("topology"),
      HdBasisCurvesTopologySchemaTokens->basis);
  static const HdDataSourceLocator type_locator(
      TfToken("basisCurves"), TfToken("topology"),
      HdBasisCurvesTopologySchemaTokens->type);
  static const HdDataSourceLocator wrap_locator(
      TfToken("basisCurves"), TfToken("topology"),
      HdBasisCurvesTopologySchemaTokens->wrap);
  static const HdDataSourceLocator vertex_counts_locator(
      TfToken("basisCurves"), TfToken("topology"),
      HdBasisCurvesTopologySchemaTokens->curveVertexCounts);

  if (*dirtyBits == HdChangeTracker::Clean) return;

  const SdfPath& id = GetId();
  HdContainerDataSourceHandle data_source =
      GetPrimDataSource(sceneDelegate, id);

  // 1. Visibility
  bool visible = GetPrimVisible(data_source);
  if ((*dirtyBits & HdChangeTracker::DirtyVisibility) && visible) {
    *dirtyBits |= HdChangeTracker::AllDirty;
  }
  SceneManager* scene_manager =
      static_cast<HdMitsubaRenderParam*>(renderParam)->GetScene();

  if (!visible) {
    RemoveFromScene(scene_manager);
    *dirtyBits = HdChangeTracker::Clean;
    return;
  }

  // 2. Transform
  GfMatrix4d transform = GetPrimTransform(data_source);

  // 3. Primvars (Points and Widths)
  if (points_.empty() || widths_.empty() ||
      HdChangeTracker::IsPrimvarDirty(*dirtyBits, id, HdTokens->points) ||
      HdChangeTracker::IsPrimvarDirty(*dirtyBits, id, HdTokens->widths)) {
    points_ = GetParam<VtVec3fArray>(data_source, points_locator);
    if (points_.empty()) {
      TF_WARN("Curves %s: missing or invalid points primvar.", id.GetText());
    }
    widths_ = GetParam<VtFloatArray>(data_source, widths_locator);
    if (widths_.empty()) {
      TF_WARN("Curves %s: missing or invalid widths primvar.", id.GetText());
    }
  }

  // 4. Topology
  std::string plugin_name = "linearcurve";
  TfToken basis = GetParam<TfToken>(data_source, basis_locator);
  TfToken type = GetParam<TfToken>(data_source, type_locator);
  const bool is_catmull_rom =
      basis == HdTokens->catmullRom && type == HdTokens->cubic;
  if ((basis == HdTokens->bspline && type == HdTokens->cubic) ||
      is_catmull_rom) {
    plugin_name = "bsplinecurve";
  }
  VtIntArray vertex_counts =
      GetParam<VtIntArray>(data_source, vertex_counts_locator);

  if (points_.empty() || widths_.empty()) {
    TF_DEBUG(HDMITSUBA_LIFECYCLE)
        .Msg("Curves %s has no points or widths. Removing from scene.\n",
             id.GetText());
    RemoveFromScene(scene_manager);
    *dirtyBits = HdChangeTracker::Clean;
    return;
  }

  bool has_negative_count = false;
  size_t total_vertices = 0;
  for (int count : vertex_counts.AsConst()) {
    if (count < 0) {
      has_negative_count = true;
    }
    total_vertices += std::max(0, count);
  }
  if (has_negative_count || total_vertices != points_.size()) {
    TF_WARN(
        "Curves %s: curveVertexCounts sum to %zu (has_negative=%d) but "
        "there are %zu points. Fix the curve topology in the USD asset. "
        "Removing from scene.",
        id.GetText(), total_vertices, has_negative_count, points_.size());
    RemoveFromScene(scene_manager);
    *dirtyBits = HdChangeTracker::Clean;
    return;
  }

  // 5. Material binding
  SdfPath material_id = GetParam<SdfPath>(data_source, material_locator);

  // 6. Convert data to CurveSpec and pass it to the scene manager.
  CurveSpec spec;
  spec.id = id;
  spec.transform = UsdToMitsubaTransform(transform);
  spec.material_id = std::move(material_id);
  spec.needs_rebuild = true;
  spec.plugin_name = std::move(plugin_name);

  // Pack control points [x, y, z, r] (lazy-evaluating widths based on layout)
  if (widths_.size() == points_.size()) {
    spec.control_points = PackControlPoints(
        points_, [&](size_t i) { return widths_.AsConst()[i] * 0.5f; });
  } else if (widths_.size() == 1) {
    float r = widths_.AsConst()[0] * 0.5f;
    spec.control_points =
        PackControlPoints(points_, [&](size_t /*i*/) { return r; });
  } else {
    float uniform_radius = CalculateMeanWidth(widths_) * 0.5f;
    spec.control_points = PackControlPoints(
        points_, [&](size_t /*i*/) { return uniform_radius; });
  }

  // Catmull-Rom: rewrite as equivalent B-spline strands.
  if (is_catmull_rom) {
    const bool pinned =
        GetParam<TfToken>(data_source, wrap_locator, HdTokens->nonperiodic) ==
        HdTokens->pinned;
    ConvertCatmullRomToBSpline(pinned, vertex_counts, spec.control_points);
  }

  // Generate segment indices
  int degree = (spec.plugin_name == "bsplinecurve") ? 3 : 1;
  int n_indices = 0;
  for (int count : vertex_counts.AsConst()) {
    n_indices += std::max(0, count - degree);
  }
  if (n_indices == 0) {
    TF_WARN("Curves %s has no valid segments. Removing from scene.",
            id.GetText());
    RemoveFromScene(scene_manager);
    *dirtyBits = HdChangeTracker::Clean;
    return;
  }
  spec.segment_indices.resize(n_indices);
  int index = 0;
  int accumulated_vertices = 0;
  for (int count : vertex_counts.AsConst()) {
    int num_segments = count - degree;
    for (int j = 0; j < num_segments; ++j, ++index) {
      spec.segment_indices[index] = accumulated_vertices + j;
    }
    accumulated_vertices += count;
  }
  scene_manager->SyncCurves(std::move(spec));
  in_scene_ = true;
  *dirtyBits = HdChangeTracker::Clean;
}

void HdMitsubaCurves::Finalize(HdRenderParam* renderParam) {
  RemoveFromScene(static_cast<HdMitsubaRenderParam*>(renderParam)->GetScene());
}

void HdMitsubaCurves::RemoveFromScene(SceneManager* scene) {
  if (in_scene_) {
    scene->RemoveShape(GetId());
    in_scene_ = false;
  }
}

HdDirtyBits HdMitsubaCurves::_PropagateDirtyBits(HdDirtyBits bits) const {
  return bits;
}

void HdMitsubaCurves::_InitRepr(const TfToken& /*reprToken*/,
                                HdDirtyBits* /*dirtyBits*/) {}

PXR_NAMESPACE_CLOSE_SCOPE

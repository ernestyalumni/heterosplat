#include "Core/CubOperations.h"
#include "IO/PlyReader.h"
#include "Kernels/Heterosplat/IntersectOffset.h"
#include "Kernels/Heterosplat/IntersectTile.h"
#include "Kernels/Heterosplat/ProjectionEWA3DGSFused.h"
#include "Kernels/Heterosplat/RasterizeToPixels3DGS.h"
#include "Kernels/Heterosplat/SphericalHarmonics.h"
#include "Normalize/Transform.h"
#include "Training/Activations.h"
#include "Viewer/ColormapDebug.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <string>
#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

namespace
{

constexpr float kPi {3.14159265358979323846f};
constexpr std::uint32_t kTileSize {16};

// ── GPU buffer ──────────────────────────────────────────────────────────────

template <typename T>
struct GpuBuffer
{
  T* ptr {nullptr};
  std::size_t count {0};

  GpuBuffer() = default;
  explicit GpuBuffer(std::size_t n) : count{n}
  {
    if (count > 0) cudaMalloc(&ptr, count * sizeof(T));
  }
  GpuBuffer(GpuBuffer&& o) noexcept : ptr{o.ptr}, count{o.count}
  { o.ptr = nullptr; o.count = 0; }
  GpuBuffer& operator=(GpuBuffer&& o) noexcept
  {
    if (this != &o) { if (ptr) cudaFree(ptr); ptr = o.ptr; count = o.count; o.ptr = nullptr; o.count = 0; }
    return *this;
  }
  ~GpuBuffer() { if (ptr) cudaFree(ptr); }
  GpuBuffer(const GpuBuffer&) = delete;
  GpuBuffer& operator=(const GpuBuffer&) = delete;

  void upload(const std::vector<T>& h) { cudaMemcpy(ptr, h.data(), h.size()*sizeof(T), cudaMemcpyHostToDevice); }
  void zero() { if (count > 0) cudaMemset(ptr, 0, count * sizeof(T)); }
  std::vector<T> download() const
  {
    std::vector<T> h(count);
    cudaMemcpy(h.data(), ptr, count*sizeof(T), cudaMemcpyDeviceToHost);
    return h;
  }
};

// ── Camera ──────────────────────────────────────────────────────────────────

struct OrbitCamera
{
  float center[3] {0, 0, 0};
  float radius {5.0f};
  float azimuth {0.0f};
  float elevation {30.0f};
  float fov {60.0f};

  float cam_pos[3] {};

  void compute_position()
  {
    const float elev {elevation * kPi / 180.0f};
    const float azim {azimuth * kPi / 180.0f};
    cam_pos[0] = center[0] + radius * std::cos(elev) * std::cos(azim);
    cam_pos[1] = center[1] + radius * std::cos(elev) * std::sin(azim);
    cam_pos[2] = center[2] + radius * std::sin(elev);
  }

  std::vector<float> viewmat() const
  {
    float fwd[3] {
      center[0] - cam_pos[0],
      center[1] - cam_pos[1],
      center[2] - cam_pos[2]};
    float norm {std::sqrt(fwd[0]*fwd[0] + fwd[1]*fwd[1] + fwd[2]*fwd[2])};
    fwd[0] /= norm; fwd[1] /= norm; fwd[2] /= norm;

    float up[3] {0, 0, 1};
    float right[3] {
      fwd[1]*up[2] - fwd[2]*up[1],
      fwd[2]*up[0] - fwd[0]*up[2],
      fwd[0]*up[1] - fwd[1]*up[0]};
    norm = std::sqrt(right[0]*right[0] + right[1]*right[1] + right[2]*right[2]);
    if (norm < 1e-6f) { right[0] = 1; right[1] = 0; right[2] = 0; norm = 1; }
    right[0] /= norm; right[1] /= norm; right[2] /= norm;

    up[0] = right[1]*fwd[2] - right[2]*fwd[1];
    up[1] = right[2]*fwd[0] - right[0]*fwd[2];
    up[2] = right[0]*fwd[1] - right[1]*fwd[0];

    const float tx {-(right[0]*cam_pos[0] + right[1]*cam_pos[1] + right[2]*cam_pos[2])};
    const float ty {-(-up[0]*cam_pos[0] + -up[1]*cam_pos[1] + -up[2]*cam_pos[2])};
    const float tz {-(fwd[0]*cam_pos[0] + fwd[1]*cam_pos[1] + fwd[2]*cam_pos[2])};

    return {
      right[0], right[1], right[2], tx,
      -up[0], -up[1], -up[2], ty,
      fwd[0], fwd[1], fwd[2], tz,
      0, 0, 0, 1};
  }

  std::vector<float> intrinsics(std::uint32_t w, std::uint32_t h) const
  {
    const float fx {static_cast<float>(w) / (2.0f * std::tan(fov * kPi / 360.0f))};
    return {fx, 0, w/2.0f, 0, fx, h/2.0f, 0, 0, 1};
  }
};

// ── Transform widget state ──────────────────────────────────────────────────

struct TransformWidget
{
  float translation[3] {0, 0, 0};
  float rotation_deg[3] {0, 0, 0};
  float scale {1.0f};
  bool dirty {true};

  void build_rotation_matrix(float* R) const
  {
    const float rx {rotation_deg[0] * kPi / 180.0f};
    const float ry {rotation_deg[1] * kPi / 180.0f};
    const float rz {rotation_deg[2] * kPi / 180.0f};

    const float cx {std::cos(rx)}, sx {std::sin(rx)};
    const float cy {std::cos(ry)}, sy {std::sin(ry)};
    const float cz {std::cos(rz)}, sz {std::sin(rz)};

    // Rz * Ry * Rx (row-major)
    R[0] = cz*cy;         R[1] = cz*sy*sx - sz*cx; R[2] = cz*sy*cx + sz*sx;
    R[3] = sz*cy;         R[4] = sz*sy*sx + cz*cx; R[5] = sz*sy*cx - cz*sx;
    R[6] = -sy;           R[7] = cy*sx;            R[8] = cy*cx;
  }
};

// ── Splat scene ─────────────────────────────────────────────────────────────

struct SplatScene
{
  std::uint32_t N {0};
  std::uint32_t K {0};
  std::uint32_t sh_degree {0};

  // Original data (immutable)
  GpuBuffer<float> d_means_orig;
  GpuBuffer<float> d_quats_orig;
  GpuBuffer<float> d_log_scales_orig;
  GpuBuffer<float> d_sh_coeffs;
  GpuBuffer<float> d_logit_opacities;

  // Working copies (transformed each frame)
  GpuBuffer<float> d_means;
  GpuBuffer<float> d_quats;
  GpuBuffer<float> d_log_scales;
  GpuBuffer<float> d_actual_scales;
  GpuBuffer<float> d_actual_opacities;
};

void load_scene(const std::string& path, SplatScene& scene, cudaStream_t stream)
{
  auto data {IO::read_gaussians_ply(path)};
  scene.N = data.num_gaussians;
  scene.sh_degree = data.sh_degree;
  scene.K = (data.sh_degree + 1) * (data.sh_degree + 1);

  scene.d_means_orig = GpuBuffer<float>(scene.N * 3);
  scene.d_quats_orig = GpuBuffer<float>(scene.N * 4);
  scene.d_log_scales_orig = GpuBuffer<float>(scene.N * 3);
  scene.d_sh_coeffs = GpuBuffer<float>(scene.N * scene.K * 3);
  scene.d_logit_opacities = GpuBuffer<float>(scene.N);

  scene.d_means_orig.upload(data.means);
  scene.d_quats_orig.upload(data.quats);
  scene.d_log_scales_orig.upload(data.scales);
  scene.d_sh_coeffs.upload(data.sh_coeffs);
  scene.d_logit_opacities.upload(data.opacities);

  scene.d_means = GpuBuffer<float>(scene.N * 3);
  scene.d_quats = GpuBuffer<float>(scene.N * 4);
  scene.d_log_scales = GpuBuffer<float>(scene.N * 3);
  scene.d_actual_scales = GpuBuffer<float>(scene.N * 3);
  scene.d_actual_opacities = GpuBuffer<float>(scene.N);

  Training::launch_sigmoid_forward(
    scene.N, scene.d_logit_opacities.ptr, scene.d_actual_opacities.ptr, stream);
  cudaStreamSynchronize(stream);
}

void apply_transform_and_activate(
  SplatScene& scene,
  const TransformWidget& widget,
  cudaStream_t stream)
{
  // Copy originals to working buffers
  cudaMemcpyAsync(scene.d_means.ptr, scene.d_means_orig.ptr,
    scene.N * 3 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(scene.d_quats.ptr, scene.d_quats_orig.ptr,
    scene.N * 4 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(scene.d_log_scales.ptr, scene.d_log_scales_orig.ptr,
    scene.N * 3 * sizeof(float), cudaMemcpyDeviceToDevice, stream);

  // Apply similarity transform if non-identity
  const bool has_transform {
    widget.translation[0] != 0 || widget.translation[1] != 0 ||
    widget.translation[2] != 0 || widget.rotation_deg[0] != 0 ||
    widget.rotation_deg[1] != 0 || widget.rotation_deg[2] != 0 ||
    widget.scale != 1.0f};

  if (has_transform)
  {
    float R[9];
    widget.build_rotation_matrix(R);

    GpuBuffer<float> d_R(9);
    GpuBuffer<float> d_t(3);
    cudaMemcpyAsync(d_R.ptr, R, 9 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_t.ptr, widget.translation, 3 * sizeof(float),
      cudaMemcpyHostToDevice, stream);

    Normalize::launch_apply_similarity_transform(
      scene.N, d_R.ptr, widget.scale, d_t.ptr,
      scene.d_means.ptr, scene.d_quats.ptr, scene.d_log_scales.ptr, stream);
  }

  // Activate scales
  Training::launch_exp_forward(
    scene.N * 3, scene.d_log_scales.ptr, scene.d_actual_scales.ptr, stream);
}

// ── Render one frame ────────────────────────────────────────────────────────

void render_frame(
  const SplatScene& scene,
  const OrbitCamera& camera,
  const std::uint32_t image_w,
  const std::uint32_t image_h,
  GpuBuffer<float>& d_render_colors,
  cudaStream_t stream)
{
  const std::uint32_t N {scene.N};
  const std::uint32_t K {scene.K};
  const std::uint32_t tile_w {(image_w + kTileSize - 1) / kTileSize};
  const std::uint32_t tile_h {(image_h + kTileSize - 1) / kTileSize};
  const std::uint32_t n_pixels {image_w * image_h};

  auto h_viewmat {camera.viewmat()};
  auto h_K {camera.intrinsics(image_w, image_h)};

  GpuBuffer<float> d_viewmat(16);
  GpuBuffer<float> d_K(9);
  cudaMemcpyAsync(d_viewmat.ptr, h_viewmat.data(), 16*sizeof(float),
    cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_K.ptr, h_K.data(), 9*sizeof(float),
    cudaMemcpyHostToDevice, stream);

  // View directions
  GpuBuffer<float> d_dirs(N * 3);
  Training::launch_compute_view_directions(
    N, scene.d_means.ptr,
    camera.cam_pos[0], camera.cam_pos[1], camera.cam_pos[2],
    d_dirs.ptr, stream);

  // Projection
  GpuBuffer<std::int32_t> d_radii(N * 2);
  GpuBuffer<float> d_means2d(N * 2);
  GpuBuffer<float> d_depths(N);
  GpuBuffer<float> d_conics(N * 3);

  Kernels::Heterosplat::launch_projection_ewa_3dgs_fused_forward(
    1, 1, N,
    scene.d_means.ptr, nullptr, scene.d_quats.ptr, scene.d_actual_scales.ptr, nullptr,
    d_viewmat.ptr, d_K.ptr,
    image_w, image_h, 0.3f, 0.01f, 1e10f, 0.0f, 0,
    d_radii.ptr, d_means2d.ptr, d_depths.ptr, d_conics.ptr,
    nullptr, stream);

  // SH
  GpuBuffer<float> d_colors(N * 3);
  Kernels::Heterosplat::launch_spherical_harmonics_forward(
    N, K, scene.sh_degree, d_dirs.ptr, scene.d_sh_coeffs.ptr, nullptr,
    d_colors.ptr, stream);

  // Intersect tile pass 1
  GpuBuffer<std::int32_t> d_tiles_per_gauss(N);
  Kernels::Heterosplat::launch_intersect_tile_forward(
    false, 1, N, 0,
    nullptr, nullptr,
    d_means2d.ptr, d_radii.ptr, d_depths.ptr,
    nullptr, nullptr, nullptr,
    kTileSize, tile_w, tile_h,
    d_tiles_per_gauss.ptr, nullptr, nullptr, stream);

  // CUB prefix sum
  GpuBuffer<std::int64_t> d_cum_tiles(N);
  Core::cub_inclusive_sum_int32_to_int64(
    N, d_tiles_per_gauss.ptr, d_cum_tiles.ptr, stream);

  std::int64_t n_isects_64 {0};
  cudaMemcpyAsync(&n_isects_64, d_cum_tiles.ptr + (N - 1),
    sizeof(std::int64_t), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  const auto n_isects {static_cast<std::uint32_t>(n_isects_64)};

  if (n_isects == 0)
  {
    d_render_colors.zero();
    return;
  }

  // Intersect tile pass 2
  GpuBuffer<std::int64_t> d_isect_ids(n_isects);
  GpuBuffer<std::int32_t> d_flatten_ids(n_isects);
  Kernels::Heterosplat::launch_intersect_tile_forward(
    false, 1, N, 0,
    nullptr, nullptr,
    d_means2d.ptr, d_radii.ptr, d_depths.ptr,
    nullptr, nullptr, d_cum_tiles.ptr,
    kTileSize, tile_w, tile_h,
    nullptr, d_isect_ids.ptr, d_flatten_ids.ptr, stream);

  // CUB radix sort
  GpuBuffer<std::int64_t> d_isect_ids_sorted(n_isects);
  GpuBuffer<std::int32_t> d_flatten_ids_sorted(n_isects);
  Core::cub_radix_sort_pairs_int64_int32(
    n_isects,
    d_isect_ids.ptr, d_isect_ids_sorted.ptr,
    d_flatten_ids.ptr, d_flatten_ids_sorted.ptr,
    0, 64, stream);

  // Intersect offset
  GpuBuffer<std::int32_t> d_tile_offsets(tile_h * tile_w);
  d_tile_offsets.zero();
  Kernels::Heterosplat::launch_intersect_offset_forward(
    n_isects, d_isect_ids_sorted.ptr,
    1, tile_w, tile_h, d_tile_offsets.ptr, stream);

  // Rasterize
  GpuBuffer<float> d_render_alphas(n_pixels);
  GpuBuffer<std::int32_t> d_last_ids(n_pixels);

  Kernels::Heterosplat::launch_rasterize_to_pixels_3dgs_forward(
    1, N, n_isects, false,
    d_means2d.ptr, d_conics.ptr, d_colors.ptr,
    scene.d_actual_opacities.ptr,
    nullptr, nullptr,
    image_w, image_h, kTileSize,
    d_tile_offsets.ptr, d_flatten_ids_sorted.ptr,
    d_render_colors.ptr, d_render_alphas.ptr, d_last_ids.ptr,
    stream);
}

// ── Second custom kernel: per-axis colormap debug ───────────────────────────
// Implemented in Viewer/ColormapDebug.cu, declared via Viewer/ColormapDebug.h

// ── Mouse/keyboard state ────────────────────────────────────────────────────

struct InputState
{
  bool dragging {false};
  double last_x {0}, last_y {0};
  bool right_dragging {false};
  double right_last_x {0}, right_last_y {0};
};

static InputState g_input;
static OrbitCamera g_camera;

void scroll_callback(GLFWwindow*, double, double y_offset)
{
  g_camera.radius *= (1.0f - 0.1f * static_cast<float>(y_offset));
  g_camera.radius = std::max(0.1f, g_camera.radius);
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int)
{
  if (ImGui::GetIO().WantCaptureMouse) return;

  if (button == GLFW_MOUSE_BUTTON_LEFT)
  {
    g_input.dragging = (action == GLFW_PRESS);
    glfwGetCursorPos(window, &g_input.last_x, &g_input.last_y);
  }
  if (button == GLFW_MOUSE_BUTTON_RIGHT)
  {
    g_input.right_dragging = (action == GLFW_PRESS);
    glfwGetCursorPos(window, &g_input.right_last_x, &g_input.right_last_y);
  }
}

void cursor_pos_callback(GLFWwindow*, double x, double y)
{
  if (ImGui::GetIO().WantCaptureMouse) return;

  if (g_input.dragging)
  {
    g_camera.azimuth -= 0.3f * static_cast<float>(x - g_input.last_x);
    g_camera.elevation += 0.3f * static_cast<float>(y - g_input.last_y);
    g_camera.elevation = std::clamp(g_camera.elevation, -89.0f, 89.0f);
    g_input.last_x = x;
    g_input.last_y = y;
  }
  if (g_input.right_dragging)
  {
    const float dx {0.005f * g_camera.radius * static_cast<float>(x - g_input.right_last_x)};
    const float dy {0.005f * g_camera.radius * static_cast<float>(y - g_input.right_last_y)};
    const float azim {g_camera.azimuth * kPi / 180.0f};
    g_camera.center[0] -= dx * std::cos(azim);
    g_camera.center[1] -= dx * std::sin(azim);
    g_camera.center[2] += dy;
    g_input.right_last_x = x;
    g_input.right_last_y = y;
  }
}

} // namespace

int main(int argc, char** argv)
{
  if (argc < 2)
  {
    std::printf("Usage: %s <input.ply> [width] [height]\n", argv[0]);
    return 1;
  }

  const std::string ply_path {argv[1]};
  std::uint32_t win_w {argc >= 3 ? static_cast<std::uint32_t>(std::atoi(argv[2])) : 1280u};
  std::uint32_t win_h {argc >= 4 ? static_cast<std::uint32_t>(std::atoi(argv[3])) : 720u};

  // ── GLFW + OpenGL ───────────────────────────────────────────────────────

  if (!glfwInit())
  {
    std::printf("Failed to initialize GLFW\n");
    return 1;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow* window {glfwCreateWindow(
    static_cast<int>(win_w), static_cast<int>(win_h),
    "heterosplat viewer", nullptr, nullptr)};
  if (!window)
  {
    std::printf("Failed to create GLFW window\n");
    glfwTerminate();
    return 1;
  }

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  glewExperimental = GL_TRUE;
  if (glewInit() != GLEW_OK)
  {
    std::printf("Failed to initialize GLEW\n");
    return 1;
  }
  std::printf("OpenGL %s\n", glGetString(GL_VERSION));

  glfwSetScrollCallback(window, scroll_callback);
  glfwSetMouseButtonCallback(window, mouse_button_callback);
  glfwSetCursorPosCallback(window, cursor_pos_callback);

  // ── Dear ImGui ────────────────────────────────────────────────────────

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 330 core");
  ImGui::StyleColorsDark();

  // ── Fullscreen quad ───────────────────────────────────────────────────

  const float quad_verts[] {
    -1, -1, 0, 0,
     1, -1, 1, 0,
     1,  1, 1, 1,
    -1, -1, 0, 0,
     1,  1, 1, 1,
    -1,  1, 0, 1};

  GLuint vao, vbo;
  glGenVertexArrays(1, &vao);
  glGenBuffers(1, &vbo);
  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(quad_verts), quad_verts, GL_STATIC_DRAW);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), nullptr);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float),
    reinterpret_cast<void*>(2*sizeof(float)));
  glEnableVertexAttribArray(1);

  const char* vert_src {
    "#version 330 core\n"
    "layout(location=0) in vec2 pos;\n"
    "layout(location=1) in vec2 uv;\n"
    "out vec2 fUV;\n"
    "void main() { fUV = uv; gl_Position = vec4(pos, 0, 1); }\n"};
  const char* frag_src {
    "#version 330 core\n"
    "in vec2 fUV;\n"
    "out vec4 color;\n"
    "uniform sampler2D tex;\n"
    "void main() { color = vec4(texture(tex, fUV).rgb, 1); }\n"};

  auto compile_shader = [](GLenum type, const char* src) -> GLuint {
    GLuint s {glCreateShader(type)};
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    return s;
  };

  GLuint prog {glCreateProgram()};
  GLuint vs {compile_shader(GL_VERTEX_SHADER, vert_src)};
  GLuint fs {compile_shader(GL_FRAGMENT_SHADER, frag_src)};
  glAttachShader(prog, vs);
  glAttachShader(prog, fs);
  glLinkProgram(prog);
  glDeleteShader(vs);
  glDeleteShader(fs);

  GLuint tex;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, win_w, win_h, 0,
    GL_RGB, GL_FLOAT, nullptr);

  // ── Load scene ────────────────────────────────────────────────────────

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  std::printf("Loading %s...\n", ply_path.c_str());
  SplatScene scene;
  load_scene(ply_path, scene, stream);
  std::printf("  %u Gaussians, SH degree %u\n", scene.N, scene.sh_degree);

  // Auto-fit camera to scene
  auto h_means {scene.d_means_orig.download()};
  float cx {0}, cy {0}, cz {0};
  float min_b[3] {1e30f, 1e30f, 1e30f};
  float max_b[3] {-1e30f, -1e30f, -1e30f};
  for (std::uint32_t i = 0; i < scene.N; ++i)
  {
    cx += h_means[i*3+0]; cy += h_means[i*3+1]; cz += h_means[i*3+2];
    for (int a = 0; a < 3; ++a)
    {
      min_b[a] = std::min(min_b[a], h_means[i*3+a]);
      max_b[a] = std::max(max_b[a], h_means[i*3+a]);
    }
  }
  cx /= scene.N; cy /= scene.N; cz /= scene.N;

  float max_dist {0};
  for (std::uint32_t i = 0; i < scene.N; ++i)
  {
    float dx {h_means[i*3+0]-cx}, dy {h_means[i*3+1]-cy}, dz {h_means[i*3+2]-cz};
    max_dist = std::max(max_dist, std::sqrt(dx*dx + dy*dy + dz*dz));
  }

  g_camera.center[0] = cx;
  g_camera.center[1] = cy;
  g_camera.center[2] = cz;
  g_camera.radius = max_dist * 2.5f;
  g_camera.compute_position();

  float bounds[6] {min_b[0], max_b[0], min_b[1], max_b[1], min_b[2], max_b[2]};

  // Render buffers
  GpuBuffer<float> d_render_colors(win_w * win_h * 3);
  std::vector<float> h_pixels(win_w * win_h * 3);

  TransformWidget widget;
  bool colormap_active {false};
  int colormap_axis {2};
  bool show_ui {true};

  std::printf("Viewer ready. Controls:\n"
    "  Left-drag: orbit | Right-drag: pan | Scroll: zoom\n"
    "  Tab: toggle UI\n");

  // ── Main loop ─────────────────────────────────────────────────────────

  auto last_time {std::chrono::steady_clock::now()};
  float fps {0.0f};

  while (!glfwWindowShouldClose(window))
  {
    glfwPollEvents();

    if (glfwGetKey(window, GLFW_KEY_TAB) == GLFW_PRESS)
      show_ui = !show_ui;

    // Handle window resize
    int fb_w, fb_h;
    glfwGetFramebufferSize(window, &fb_w, &fb_h);
    if (fb_w > 0 && fb_h > 0 &&
        (static_cast<std::uint32_t>(fb_w) != win_w ||
         static_cast<std::uint32_t>(fb_h) != win_h))
    {
      win_w = static_cast<std::uint32_t>(fb_w);
      win_h = static_cast<std::uint32_t>(fb_h);
      glBindTexture(GL_TEXTURE_2D, tex);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, win_w, win_h, 0,
        GL_RGB, GL_FLOAT, nullptr);
      d_render_colors = GpuBuffer<float>(win_w * win_h * 3);
      h_pixels.resize(win_w * win_h * 3);
    }

    // ── ImGui ───────────────────────────────────────────────────────────

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if (show_ui)
    {
      ImGui::Begin("Transform");
      bool changed {false};
      changed |= ImGui::SliderFloat3("Translation", widget.translation, -5.0f, 5.0f);
      changed |= ImGui::SliderFloat3("Rotation (deg)", widget.rotation_deg, -180.0f, 180.0f);
      changed |= ImGui::SliderFloat("Scale", &widget.scale, 0.1f, 5.0f);
      if (ImGui::Button("Reset"))
      {
        widget.translation[0] = widget.translation[1] = widget.translation[2] = 0;
        widget.rotation_deg[0] = widget.rotation_deg[1] = widget.rotation_deg[2] = 0;
        widget.scale = 1.0f;
        changed = true;
      }
      widget.dirty |= changed;
      ImGui::End();

      ImGui::Begin("Camera");
      ImGui::SliderFloat("FOV", &g_camera.fov, 10.0f, 120.0f);
      ImGui::Text("Azimuth: %.1f  Elevation: %.1f", g_camera.azimuth, g_camera.elevation);
      ImGui::Text("Radius: %.2f", g_camera.radius);
      ImGui::Text("FPS: %.1f", fps);
      ImGui::Text("Gaussians: %u", scene.N);
      ImGui::End();

      ImGui::Begin("Debug");
      bool cm_changed {ImGui::Checkbox("Colormap by axis", &colormap_active)};
      cm_changed |= ImGui::RadioButton("X", &colormap_axis, 0); ImGui::SameLine();
      cm_changed |= ImGui::RadioButton("Y", &colormap_axis, 1); ImGui::SameLine();
      cm_changed |= ImGui::RadioButton("Z", &colormap_axis, 2);
      if (cm_changed) widget.dirty = true;
      ImGui::End();
    }

    // ── Render ──────────────────────────────────────────────────────────

    g_camera.compute_position();

    apply_transform_and_activate(scene, widget, stream);

    if (colormap_active)
    {
      Viewer::launch_colormap_per_axis(
        scene.N, scene.d_means.ptr,
        bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5],
        static_cast<std::uint32_t>(colormap_axis),
        scene.K, scene.d_sh_coeffs.ptr, stream);
    }

    render_frame(scene, g_camera, win_w, win_h, d_render_colors, stream);
    cudaStreamSynchronize(stream);

    widget.dirty = false;

    // Download and upload to GL texture
    cudaMemcpy(h_pixels.data(), d_render_colors.ptr,
      win_w * win_h * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, win_w, win_h,
      GL_RGB, GL_FLOAT, h_pixels.data());

    // Draw fullscreen quad
    glViewport(0, 0, win_w, win_h);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(prog);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    // Draw ImGui
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);

    // FPS
    auto now {std::chrono::steady_clock::now()};
    float dt {std::chrono::duration<float>(now - last_time).count()};
    fps = 1.0f / std::max(dt, 1e-6f);
    last_time = now;
  }

  // ── Cleanup ───────────────────────────────────────────────────────────

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glDeleteTextures(1, &tex);
  glDeleteBuffers(1, &vbo);
  glDeleteVertexArrays(1, &vao);
  glDeleteProgram(prog);

  cudaStreamDestroy(stream);

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}

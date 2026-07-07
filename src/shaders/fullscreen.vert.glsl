#version 300 es
// Fullscreen triangle generated purely from gl_VertexID — no vertex buffers.
// Vertices land at (-1,-1), (3,-1), (-1,3); the triangle covers the viewport
// and the GPU clips away the overhang. This avoids the diagonal seam of a
// two-triangle quad.
void main() {
    vec2 pos = vec2(
        float((gl_VertexID << 1) & 2) * 2.0 - 1.0,
        float(gl_VertexID & 2) * 2.0 - 1.0
    );
    gl_Position = vec4(pos, 0.0, 1.0);
}

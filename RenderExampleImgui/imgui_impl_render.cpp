#include <Render/Render.h>

#include "imgui_impl_render.h"

#include <memory>

using namespace tpr;

static RenderFormat g_TargetFormat;

static VertexShader_t g_VS = VertexShader_t::INVALID;
static PixelShader_t g_PS = PixelShader_t::INVALID;
static Texture_t g_FontTexture = Texture_t::INVALID;
static ShaderResourceView_t g_FontSrv = ShaderResourceView_t::INVALID;
static RootSignature_t g_RootSignature = RootSignature_t::INVALID;
static GraphicsPipelineState_t g_PSO = GraphicsPipelineState_t::INVALID;
static int g_VertexBufferSize = 5000;
static int g_IndexBufferSize = 10000;

// Not thread safe at all but allows us to accumulate each frames draw data without reallocating huge arrays.
static std::unique_ptr<ImDrawVert[]> g_VertexAlloc;
static std::unique_ptr<ImDrawIdx[]> g_IndexAlloc;

struct ImRenderFrameData
{
    DynamicBuffer_t VertexBuffer = DynamicBuffer_t::INVALID;
    DynamicBuffer_t IndexBuffer = DynamicBuffer_t::INVALID;
    DynamicBuffer_t ViewBuffer = DynamicBuffer_t::INVALID;
};

// Forward Declarations
static void ImGui_ImplRender_InitPlatformInterface();
static void ImGui_ImplRender_ShutdownPlatformInterface();

static void ImGui_ImplRender_SetupRenderState(ImDrawData* draw_data, tpr::CommandList* cl, DynamicBuffer_t vb, DynamicBuffer_t ib, DynamicBuffer_t cb)
{
    // Setup viewport
    Viewport vp{};
    ZeroMemory(&vp, sizeof(vp));
    vp.width = draw_data->DisplaySize.x;
    vp.height = draw_data->DisplaySize.y;
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;
    vp.topLeftX = vp.topLeftY = 0;

    cl->SetViewports(&vp, 1);

    // Setup buffers
    unsigned int stride = sizeof(ImDrawVert);
    unsigned int offset = 0;
    cl->SetVertexBuffers(0, 1, &vb, &stride, &offset);
    cl->SetIndexBuffer(ib, sizeof(ImDrawIdx) == 2 ? RenderFormat::R16_UINT : RenderFormat::R32_UINT, 0);

    if (Render_IsBindless())
    {
        cl->SetGraphicsRootCBV(0, cb);
    }
    else
    {
        cl->BindVertexCBVs(0, 1, &cb);
    }    
}

IMGUI_IMPL_API ImRenderFrameData* ImGui_ImplRender_PrepareFrameData(ImDrawData* draw_data)
{
    // Avoid rendering when minimized
    if (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f)
        return nullptr;

    if (!g_VertexAlloc || g_VertexBufferSize < draw_data->TotalVtxCount)
    {
        g_VertexBufferSize = draw_data->TotalVtxCount + 5000;
        g_VertexAlloc = std::make_unique<ImDrawVert[]>(g_VertexBufferSize);
    }

    if (!g_IndexAlloc || g_IndexBufferSize < draw_data->TotalIdxCount)
    {
        g_IndexBufferSize = draw_data->TotalIdxCount + 10000;
        g_IndexAlloc = std::make_unique<ImDrawIdx[]>(g_IndexBufferSize);
    }

    ImDrawVert* vtx_dst = g_VertexAlloc.get();
    ImDrawIdx* idx_dst = g_IndexAlloc.get();
    for (int n = 0; n < draw_data->CmdListsCount; n++)
    {
        const ImDrawList* cmd_list = draw_data->CmdLists[n];
        memcpy(vtx_dst, cmd_list->VtxBuffer.Data, cmd_list->VtxBuffer.Size * sizeof(ImDrawVert));
        memcpy(idx_dst, cmd_list->IdxBuffer.Data, cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx));
        vtx_dst += cmd_list->VtxBuffer.Size;
        idx_dst += cmd_list->IdxBuffer.Size;
    }

    DynamicBuffer_t vbuf = CreateDynamicVertexBuffer(g_VertexAlloc.get(), g_VertexBufferSize * sizeof(ImDrawVert));
    DynamicBuffer_t ibuf = CreateDynamicIndexBuffer(g_IndexAlloc.get(), g_IndexBufferSize * sizeof(ImDrawIdx));

    // Setup orthographic projection matrix into our constant buffer
    // Our visible imgui space lies from draw_data->DisplayPos (top left) to draw_data->DisplayPos+data_data->DisplaySize (bottom right). DisplayPos is (0,0) for single viewport apps.

    float L = draw_data->DisplayPos.x;
    float R = draw_data->DisplayPos.x + draw_data->DisplaySize.x;
    float T = draw_data->DisplayPos.y;
    float B = draw_data->DisplayPos.y + draw_data->DisplaySize.y;
    float mvp[4][4] =
    {
        { 2.0f/(R-L),   0.0f,           0.0f,       0.0f },
        { 0.0f,         2.0f/(T-B),     0.0f,       0.0f },
        { 0.0f,         0.0f,           0.5f,       0.0f },
        { (R+L)/(L-R),  (T+B)/(B-T),    0.5f,       1.0f },
    };
    DynamicBuffer_t cbuf = CreateDynamicConstantBuffer(mvp, sizeof(mvp));

    ImRenderFrameData* frameData = new ImRenderFrameData;
    frameData->VertexBuffer = vbuf;
    frameData->IndexBuffer = ibuf;
    frameData->ViewBuffer = cbuf;

    return frameData;
}

// Render function
// (this used to be set in io.RenderDrawListsFn and called by ImGui::Render(), but you can now call this directly from your main loop)
void ImGui_ImplRender_RenderDrawData(ImRenderFrameData* frame_data, ImDrawData* draw_data, CommandList* cl)
{
    // Avoid rendering when minimized
    if (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f)
        return;

    GraphicsPipelineState_t prevPSO = cl->GetPreviousPSO();

    if (Render_IsBindless())
    {
        cl->SetGraphicsRootDescriptorTable(2u);
    }

    ImGui_ImplRender_SetupRenderState(draw_data, cl, frame_data->VertexBuffer, frame_data->IndexBuffer, frame_data->ViewBuffer);

    // Render command lists
    // (Because we merged all buffers into a single one, we maintain our own offset into them)
    int global_idx_offset = 0;
    int global_vtx_offset = 0;
    ImVec2 clip_off = draw_data->DisplayPos;
    for (int n = 0; n < draw_data->CmdListsCount; n++)
    {
        const ImDrawList* cmd_list = draw_data->CmdLists[n];
        for (int cmd_i = 0; cmd_i < cmd_list->CmdBuffer.Size; cmd_i++)
        {
            const ImDrawCmd* pcmd = &cmd_list->CmdBuffer[cmd_i];
            if (pcmd->UserCallback != NULL)
            {
                // User callback, registered via ImDrawList::AddCallback()
                // (ImDrawCallback_ResetRenderState is a special callback value used by the user to request the renderer to reset render state.)
                if (pcmd->UserCallback == ImDrawCallback_ResetRenderState)
                    ImGui_ImplRender_SetupRenderState(draw_data, cl, frame_data->VertexBuffer, frame_data->IndexBuffer, frame_data->ViewBuffer);
                else
                    pcmd->UserCallback(cmd_list, pcmd);
            }
            else
            {
                // Project scissor/clipping rectangles into framebuffer space
                ImVec2 clip_min(pcmd->ClipRect.x - clip_off.x, pcmd->ClipRect.y - clip_off.y);
                ImVec2 clip_max(pcmd->ClipRect.z - clip_off.x, pcmd->ClipRect.w - clip_off.y);
                if (clip_max.x <= clip_min.x || clip_max.y <= clip_min.y)
                    continue;

                // Apply scissor/clipping rectangle
                ScissorRect r;
                r.left = (uint32_t)(clip_min.x);
                r.top = (uint32_t)(clip_min.y);
                r.right = (uint32_t)(clip_max.x);
                r.bottom = (uint32_t)(clip_max.y);

                cl->SetScissors(&r, 1);

                ShaderResourceView_t srv = (ShaderResourceView_t)(intptr_t)pcmd->TextureId;

                if (srv != ShaderResourceView_t::INVALID)
                    cl->SetPipelineState(g_PSO);

                if (Render_IsBindless())
                {
                    cl->SetGraphicsRootValue(1, 0u, GetDescriptorIndex(srv));
                }
                else
                {
                    cl->BindPixelSRVs(0, 1, &srv);
                }

                cl->DrawIndexedInstanced(pcmd->ElemCount, 1, pcmd->IdxOffset + global_idx_offset, pcmd->VtxOffset + global_vtx_offset, 0);
            }
        }
        global_idx_offset += cmd_list->IdxBuffer.Size;
        global_vtx_offset += cmd_list->VtxBuffer.Size;
    }

    cl->SetPipelineState(prevPSO);
}

IMGUI_IMPL_API void ImGui_ImplRender_ReleaseFrameData(ImRenderFrameData* frame_data)
{
    if(frame_data)
        delete frame_data;
}

static void ImGui_ImplRender_CreateFontsTexture()
{
    // Build texture atlas
    ImGuiIO& io = ImGui::GetIO();
    unsigned char* pixels;
    int width, height;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);

    // Upload texture to graphics system
    {
        TextureCreateDesc desc = {};
        desc.Width = width;
        desc.Height = height;
        desc.Format = RenderFormat::R8G8B8A8_UNORM;
        desc.Flags = RenderResourceFlags::SRV;

        MipData mip{pixels, desc.Format, desc.Width, desc.Height };

        desc.Data = &mip;

        g_FontTexture = CreateTexture(desc);
        g_FontSrv = CreateTextureSRV(g_FontTexture, desc.Format, TextureDimension::TEX2D, 1u, 1u);
    }   

    io.Fonts->TexID = (ImTextureID)(g_FontSrv);
}

bool ImGui_ImplRender_CreateDeviceObjects()
{
    if (!Render_Initialised())
        return false;

    if(g_FontTexture != Texture_t::INVALID)
        ImGui_ImplRender_InvalidateDeviceObjects();

    const std::string shaderPath = "Shaders/ImGui.hlsl";

    g_VS = CreateVertexShader(shaderPath.c_str(), {});
    g_PS = CreatePixelShader(shaderPath.c_str(), {});

    {
        RootSignatureDesc rsDesc = {};
        rsDesc.Flags = RootSignatureFlags::ALLOW_INPUT_LAYOUT;
        rsDesc.Slots.push_back(RootSignatureSlot::CBVSlot(0u, 0u));
        rsDesc.Slots.push_back(RootSignatureSlot::ConstantsSlot(4u, 1u));
        rsDesc.Slots.push_back(RootSignatureSlot::DescriptorTableSlot(0u, 0u, RootSignatureDescriptorTableType::SRV));
        rsDesc.GlobalSamplers.resize(1u);
        rsDesc.GlobalSamplers[0].AddressModeUVW(SamplerAddressMode::WRAP).FilterModeMinMagMip(SamplerFilterMode::POINT);

        g_RootSignature = CreateRootSignature(rsDesc);
    }

    {
        GraphicsPipelineStateDesc pipeDesc = {};
        pipeDesc.RasterizerDesc(PrimitiveTopologyType::TRIANGLE, FillMode::SOLID, CullMode::NONE)
                .DepthDesc(false)
                .TargetBlendDesc({ g_TargetFormat }, { BlendMode::Default() }, RenderFormat::UNKNOWN)
                .VertexShader(g_VS)
                .PixelShader(g_PS)
                .RootSignature(g_RootSignature);

        InputElementDesc local_layout[] =
        {
            { "POSITION", 0, RenderFormat::R32G32_FLOAT,   0, (size_t)(&((ImDrawVert*)0)->pos), InputClassification::PER_VERTEX, 0 },
            { "TEXCOORD", 0, RenderFormat::R32G32_FLOAT,   0, (size_t)(&((ImDrawVert*)0)->uv),  InputClassification::PER_VERTEX, 0 },
            { "COLOR",    0, RenderFormat::R8G8B8A8_UNORM, 0, (size_t)(&((ImDrawVert*)0)->col), InputClassification::PER_VERTEX, 0 },
        };

        g_PSO = CreateGraphicsPipelineState(pipeDesc, local_layout, 3);
    }

    ImGui_ImplRender_CreateFontsTexture();

    return true;
}

RootSignature_t ImGui_ImplRender_GetRootSignature()
{
    return g_RootSignature;
}

bool ImGui_ImplRender_Init(RenderFormat targetFormat)
{
    ImGuiIO& io = ImGui::GetIO();
    io.BackendRendererName = "imgui_impl_render";
    io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;  // We can honor the ImDrawCmd::VtxOffset field, allowing for large meshes.

    io.BackendFlags |= ImGuiBackendFlags_RendererHasViewports;  // We can create multi-viewports on the Renderer side (optional)

    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        ImGui_ImplRender_InitPlatformInterface();

    g_TargetFormat = targetFormat;

    return true;
}

void ImGui_ImplRender_Shutdown()
{
    ImGui_ImplRender_ShutdownPlatformInterface();
    ImGui_ImplRender_InvalidateDeviceObjects();
}

void ImGui_ImplRender_NewFrame()
{
    if (g_FontTexture == Texture_t::INVALID)
        ImGui_ImplRender_CreateDeviceObjects();
}

void ImGui_ImplRender_InvalidateDeviceObjects()
{
    if (!Render_Initialised())
        return;

    RenderRelease(g_FontTexture); g_FontTexture = {};
    RenderRelease(g_FontSrv);     g_FontSrv = {};
    RenderRelease(g_PSO); g_PSO = {};
}

static void ImGui_ImplRender_CreateWindow(ImGuiViewport* viewport)
{
    // PlatformHandleRaw should always be a HWND, whereas PlatformHandle might be a higher-level handle (e.g. GLFWWindow*, SDL_Window*).
    // Some back-end will leave PlatformHandleRaw NULL, in which case we assume PlatformHandle will contain the HWND.
    HWND hwnd = viewport->PlatformHandleRaw ? (HWND)viewport->PlatformHandleRaw : (HWND)viewport->PlatformHandle;
    IM_ASSERT(hwnd != 0);

    RenderView* rv = CreateRenderView((intptr_t)hwnd);

    if (rv)
        rv->Resize((uint32_t)viewport->Size.x, (uint32_t)viewport->Size.y);

    viewport->RendererUserData = rv;
}

static void ImGui_ImplRender_DestroyWindow(ImGuiViewport* viewport)
{
    if (RenderView* data = (RenderView*)viewport->RendererUserData)
        delete data;

    viewport->RendererUserData = nullptr;
}

static void ImGui_ImplRender_SetWindowSize(ImGuiViewport* viewport, ImVec2 size)
{
    if (RenderView* data = (RenderView*)viewport->RendererUserData)
        data->Resize((uint32_t)size.x, (uint32_t)size.y);
}

static void ImGui_ImplRender_RenderWindow(ImGuiViewport* viewport, void*)
{
    RenderView* view = (RenderView*)viewport->RendererUserData;

    view->Sync();

    Render_BeginFrame();

    ImRenderFrameData* frameData = ImGui_ImplRender_PrepareFrameData(viewport->DrawData);

    Render_BeginRenderFrame();

    CommandListSubmissionGroup clGroup(CommandListType::GRAPHICS);

    CommandList* cl = clGroup.CreateCommandList();

    UploadBuffers(cl);

    cl->TransitionResource(view->GetCurrentBackBufferTexture(), ResourceTransitionState::PRESENT, ResourceTransitionState::RENDER_TARGET);

    // Bind and clear targets
    {
        RenderTargetView_t backBufferRtv = view->GetCurrentBackBufferRTV();

        constexpr float DefaultClearCol[4] = { 0.0f, 0.0f, 0.2f, 0.0f };

        cl->ClearRenderTarget(backBufferRtv, DefaultClearCol);

        cl->SetRenderTargets(&backBufferRtv, 1, DepthStencilView_t::INVALID);
    }

    cl->SetRootSignature(ImGui_ImplRender_GetRootSignature());

    ImGui_ImplRender_RenderDrawData(frameData, viewport->DrawData, cl);

    ImGui_ImplRender_ReleaseFrameData(frameData);

    cl->TransitionResource(view->GetCurrentBackBufferTexture(), ResourceTransitionState::RENDER_TARGET, ResourceTransitionState::PRESENT);

    clGroup.Submit();

    Render_EndFrame();
}

static void ImGui_ImplRender_SwapBuffers(ImGuiViewport* viewport, void*)
{
    if (RenderView* data = (RenderView*)viewport->RendererUserData)
        data->Present(true, false);
}

static void ImGui_ImplRender_InitPlatformInterface()
{
    ImGuiPlatformIO& platform_io = ImGui::GetPlatformIO();
    platform_io.Renderer_CreateWindow = ImGui_ImplRender_CreateWindow;
    platform_io.Renderer_DestroyWindow = ImGui_ImplRender_DestroyWindow;
    platform_io.Renderer_SetWindowSize = ImGui_ImplRender_SetWindowSize;
    platform_io.Renderer_RenderWindow = ImGui_ImplRender_RenderWindow;
    platform_io.Renderer_SwapBuffers = ImGui_ImplRender_SwapBuffers;
}

static void ImGui_ImplRender_ShutdownPlatformInterface()
{
    ImGui::DestroyPlatformWindows();
}
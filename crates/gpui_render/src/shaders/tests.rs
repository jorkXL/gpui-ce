use super::*;

#[test]
fn validates_subpixel_shader() {
    let generated = subpixel_sprite::WGSL_SOURCE.wgsl_source().unwrap();
    wgsl_rs::validate_wgsl_source(&format!("enable dual_source_blending;\n{generated}")).unwrap();
}

#[test]
fn build_generated_shaders_match_rust_sources() {
    assert_eq!(
        base::WGSL_SOURCE.wgsl_source().unwrap(),
        crate::artifacts::BASE_WGSL
    );
    let subpixel = subpixel_sprite::WGSL_SOURCE.wgsl_source().unwrap();
    assert_eq!(
        format!("enable dual_source_blending;\n{subpixel}"),
        crate::artifacts::SUBPIXEL_DUAL_SOURCE_WGSL
    );
}

#[test]
fn shader_interface_matches_generated_sources() {
    let standard_source = base::WGSL_SOURCE.wgsl_source().unwrap();
    let subpixel_source = subpixel_sprite::WGSL_SOURCE.wgsl_source().unwrap();
    for pipeline in interface::ALL {
        let source = if pipeline.label == interface::SUBPIXEL_SPRITES.label {
            &subpixel_source
        } else {
            &standard_source
        };
        assert!(
            source.contains(&format!("fn {}(", pipeline.vertex_entry)),
            "missing vertex entry point {}",
            pipeline.vertex_entry,
        );
        assert!(
            source.contains(&format!("fn {}(", pipeline.fragment_entry)),
            "missing fragment entry point {}",
            pipeline.fragment_entry,
        );
    }

    assert_eq!(common::GLOBALS.group, interface::GLOBAL_BIND_GROUP);
    assert_eq!(common::GLOBALS.binding, interface::GLOBAL_UNIFORMS_BINDING);
    assert_eq!(
        common::FONT_RASTERIZATION.group,
        interface::GLOBAL_BIND_GROUP
    );
    assert_eq!(
        common::FONT_RASTERIZATION.binding,
        interface::FONT_RASTERIZATION_BINDING
    );
    assert_eq!(quad::QUADS.group(), interface::DATA_BIND_GROUP);
    assert_eq!(quad::QUADS.binding(), interface::DATA_BUFFER_BINDING);
}

fn vertex_output_shape(module: &naga::Module, entry_name: &str) -> (usize, u32) {
    let entry = module
        .entry_points
        .iter()
        .find(|entry| entry.name == entry_name)
        .unwrap_or_else(|| panic!("missing shader entry point {entry_name}"));
    let result = entry
        .function
        .result
        .as_ref()
        .unwrap_or_else(|| panic!("shader entry point {entry_name} has no output"));
    let naga::TypeInner::Struct { members, .. } = &module.types[result.ty].inner else {
        panic!("shader entry point {entry_name} must return a struct");
    };

    members
        .iter()
        .filter(|member| matches!(member.binding, Some(naga::Binding::Location { .. })))
        .fold((0, 0), |(locations, components), member| {
            let member_components = match module.types[member.ty].inner {
                naga::TypeInner::Scalar(_) => 1,
                naga::TypeInner::Vector { size, .. } => u32::from(size),
                ref ty => panic!("unsupported stage output type {ty:?} in {entry_name}"),
            };
            (locations + 1, components + member_components)
        })
}

#[test]
fn ordinary_vertex_interfaces_stay_within_compact_budgets() {
    let module = naga::front::wgsl::parse_str(&base::WGSL_SOURCE.wgsl_source().unwrap()).unwrap();
    for (entry, maximum_shape) in [
        ("vertex_quad", (8, 29)),
        ("vertex_shadow", (5, 17)),
        ("vertex_polychrome_sprite", (3, 7)),
        ("vertex_blur_composite", (2, 6)),
    ] {
        let actual_shape = vertex_output_shape(&module, entry);
        assert!(
            actual_shape.0 <= maximum_shape.0 && actual_shape.1 <= maximum_shape.1,
            "{entry} uses {actual_shape:?}, exceeding {maximum_shape:?} (locations, components)"
        );
    }
}

#[test]
fn scene_instance_storage_sizes_are_bounded() {
    for (name, actual, maximum) in [
        ("Quad", std::mem::size_of::<gpui::Quad>(), 232),
        ("Shadow", std::mem::size_of::<gpui::Shadow>(), 168),
        (
            "PolychromeSprite",
            std::mem::size_of::<gpui::PolychromeSprite>(),
            96,
        ),
    ] {
        assert!(
            actual <= maximum,
            "{name} is {actual} bytes, exceeding its {maximum}-byte storage budget"
        );
    }
}

#[test]
fn shader_buffer_layouts_match_host_layouts() {
    let modules = [
        crate::artifacts::BASE_WGSL,
        crate::artifacts::SUBPIXEL_DUAL_SOURCE_WGSL,
    ]
    .map(|source| naga::front::wgsl::parse_str(source).unwrap());

    for layouts in [gpui::SCENE_BUFFER_LAYOUTS, interface::RENDER_BUFFER_LAYOUTS] {
        for layout in layouts {
            let ty = modules
                .iter()
                .find_map(|module| {
                    module
                        .types
                        .iter()
                        .find_map(|(_, ty)| (ty.name.as_deref() == Some(layout.name)).then_some(ty))
                })
                .unwrap_or_else(|| panic!("missing shader ABI type {}", layout.name));
            let naga::TypeInner::Struct { members, span } = &ty.inner else {
                panic!("shader ABI type {} must be a struct", layout.name);
            };

            assert_eq!(
                *span as usize, layout.size,
                "shader ABI size: {}",
                layout.name
            );
            assert_eq!(
                members.len(),
                layout.fields.len(),
                "shader ABI fields: {}",
                layout.name
            );
            for (member, (name, offset)) in members.iter().zip(layout.fields) {
                assert_eq!(
                    member.name.as_deref(),
                    Some(*name),
                    "shader ABI field: {}",
                    layout.name
                );
                assert_eq!(
                    member.offset as usize, *offset,
                    "shader ABI offset: {}.{name}",
                    layout.name
                );
            }
        }
    }
}

#[test]
fn shader_discriminants_match_scene_types() {
    assert_eq!(common::ShaderBool::Disabled as u32, 0);
    assert_eq!(common::ShaderBool::Enabled as u32, 1);
    assert_eq!(
        common::BackgroundTag::Solid as u32,
        gpui::BackgroundTag::Solid as u32
    );
    assert_eq!(
        common::BackgroundTag::LinearGradient as u32,
        gpui::BackgroundTag::LinearGradient as u32
    );
    assert_eq!(
        common::BackgroundTag::PatternSlash as u32,
        gpui::BackgroundTag::PatternSlash as u32
    );
    assert_eq!(
        common::BackgroundTag::Checkerboard as u32,
        gpui::BackgroundTag::Checkerboard as u32
    );
    assert_eq!(
        common::ColorSpace::Srgb as u32,
        gpui::ColorSpace::Srgb as u32
    );
    assert_eq!(
        common::ColorSpace::Oklab as u32,
        gpui::ColorSpace::Oklab as u32
    );
    assert_eq!(
        common::BorderStyle::Solid as u32,
        gpui::BorderStyle::Solid as u32
    );
    assert_eq!(
        common::BorderStyle::Dashed as u32,
        gpui::BorderStyle::Dashed as u32
    );
}

#[test]
fn linear_gradients_preserve_native_dithering() {
    use common::*;
    use wgsl_rs::std::*;

    let color = Hsla {
        h: 0.0,
        s: 0.0,
        l: 0.5,
        a: 0.5,
    };
    let background = Background {
        tag: BackgroundTag::LinearGradient,
        color_space: ColorSpace::Srgb,
        solid: color,
        gradient_angle_or_pattern_height: 90.0,
        colors: [
            LinearColorStop {
                color,
                percentage: 0.0,
            },
            LinearColorStop {
                color,
                percentage: 1.0,
            },
        ],
        padding: 0,
    };
    let bounds = Bounds {
        origin: vec2f(0.0, 0.0),
        size: vec2f(100.0, 100.0),
    };
    let paint = Paint::new(background, bounds);
    let prepared = prepare_paint(paint);
    let first = paint_color(paint, vec2f(10.0, 10.0), prepared);
    let second = paint_color(paint, vec2f(11.0, 10.0), prepared);

    assert_ne!(first.w, second.w);
}

/// What a `_UNORM_SRGB` render target stores after blending premultiplied `source` over
/// `stored` (ONE / INV_SRC_ALPHA): it decodes, blends in linear light, encodes and rounds.
fn blend_through_srgb_view(source: wgsl_rs::std::Vec4f, stored: f32) -> f32 {
    use common::*;
    use wgsl_rs::std::*;
    let blended = source.x + srgb_to_linear(vec3f(stored, stored, stored)).x * (1.0 - source.w);
    (linear_to_srgb(vec3f(blended, blended, blended)).x * 255.0).round() / 255.0
}

#[test]
fn linear_path_output_keeps_opaque_colours_and_mixes_edges_in_light() {
    use common::*;
    use path_rasterization::premultiply_linear;
    use wgsl_rs::std::*;

    // A fully covered opaque path lands on its own 8-bit value over any destination.
    for value in 0..=255u8 {
        let encoded = f32::from(value) / 255.0;
        let source = premultiply_linear(vec4f(encoded, encoded, encoded, 1.0), 1.0);
        for stored in [0.0, 0.5, 1.0] {
            assert_eq!(blend_through_srgb_view(source, stored), encoded, "{value}");
        }
    }

    // A white edge split over two pixels at coverage c and 1 - c, over black: in linear
    // light the pair shows one pixel of white wherever the edge falls. Blended as encoded
    // values (the UNORM view), half coverage shows 21% of white's light, not 50%.
    let light = |encoded: f32| srgb_to_linear(vec3f(encoded, encoded, encoded)).x;
    for step in 0..=16 {
        let coverage = step as f32 / 16.0;
        let pair: f32 = [coverage, 1.0 - coverage]
            .map(|c| {
                light(blend_through_srgb_view(
                    premultiply_linear(vec4f(1.0, 1.0, 1.0, 1.0), c),
                    0.0,
                ))
            })
            .iter()
            .sum();
        assert!((pair - 1.0).abs() < 0.01, "coverage {coverage}: {pair}");
    }
    let encoded_half = premultiply(vec4f(1.0, 1.0, 1.0, 1.0), 0.5).x;
    assert!((2.0 * light(encoded_half) - 0.43).abs() < 0.01);
}

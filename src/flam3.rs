use std::fs;
use std::path::Path;

use quick_xml::Reader;
use quick_xml::events::Event;
use rand::Rng;
use rand::prelude::IndexedRandom;

use crate::genome::{FlameGenome, FlameTransform, GlobalParams, KifsParams};

const VARIATION_PARAM_NAMES: [&str; 8] = [
    "rings2_val",
    "blob_low",
    "blob_high",
    "blob_waves",
    "julian_power",
    "julian_dist",
    "ngon_sides",
    "ngon_corners",
];

const VARIATION_NAMES: [&str; 26] = [
    "linear",
    "sinusoidal",
    "spherical",
    "swirl",
    "horseshoe",
    "handkerchief",
    "julia",
    "polar",
    "disc",
    "rings",
    "bubble",
    "fisheye",
    "exponential",
    "spiral",
    "diamond",
    "bent",
    "waves",
    "popcorn",
    "fan",
    "eyefish",
    "cross",
    "tangent",
    "cosine",
    "blob",
    "noise",
    "curl",
];

pub struct Flam3File {
    pub flames: Vec<FlameGenome>,
}

impl Flam3File {
    pub fn parse(xml: &str) -> Result<Self, String> {
        let mut reader = Reader::from_str(xml);
        let mut flames = Vec::new();
        let mut current_flame: Option<FlameBuilder> = None;

        loop {
            match reader.read_event() {
                Ok(Event::Start(ref e)) | Ok(Event::Empty(ref e)) => {
                    let tag = String::from_utf8_lossy(e.name().as_ref()).to_string();

                    match tag.as_str() {
                        "flame" => {
                            let mut builder = FlameBuilder::new();
                            for attr in e.attributes().flatten() {
                                let key = String::from_utf8_lossy(attr.key.as_ref()).to_string();
                                let val = String::from_utf8_lossy(&attr.value).to_string();
                                match key.as_str() {
                                    "name" => builder.name = val,
                                    "symmetry" => {
                                        builder.symmetry = val.parse().unwrap_or(1);
                                    }
                                    _ => {}
                                }
                            }
                            current_flame = Some(builder);
                        }
                        "xform" => {
                            if let Some(ref mut flame) = current_flame {
                                flame.transforms.push(parse_xform(e));
                            }
                        }
                        "finalxform" => {
                            if let Some(ref mut flame) = current_flame {
                                flame.final_transform = Some(parse_xform(e));
                            }
                        }
                        "palette" => {
                            // Palette content comes as text inside the element
                            // We'll handle it in the Text event
                        }
                        _ => {}
                    }
                }
                Ok(Event::Text(ref t)) => {
                    if let Some(ref mut flame) = current_flame {
                        let text = String::from_utf8_lossy(t.as_ref()).to_string();
                        let trimmed = text.trim();
                        if !trimmed.is_empty() && trimmed.len() > 10 {
                            // Likely palette hex data
                            let palette = parse_palette(trimmed);
                            if palette.len() >= 2 {
                                flame.palette = Some(palette);
                            }
                        }
                    }
                }
                Ok(Event::End(ref e)) => {
                    let tag = String::from_utf8_lossy(e.name().as_ref()).to_string();
                    if tag == "flame"
                        && let Some(builder) = current_flame.take()
                    {
                        flames.push(builder.build());
                    }
                }
                Ok(Event::Eof) => break,
                Err(e) => return Err(format!("XML parse error: {e}")),
                _ => {}
            }
        }

        Ok(Flam3File { flames })
    }
}

struct FlameBuilder {
    name: String,
    symmetry: i32,
    transforms: Vec<FlameTransform>,
    final_transform: Option<FlameTransform>,
    palette: Option<Vec<[f32; 3]>>,
}

impl FlameBuilder {
    fn new() -> Self {
        Self {
            name: String::new(),
            symmetry: 1,
            transforms: Vec::new(),
            final_transform: None,
            palette: None,
        }
    }

    fn build(self) -> FlameGenome {
        FlameGenome {
            name: if self.name.is_empty() {
                "imported_flame".to_string()
            } else {
                self.name
            },
            global: GlobalParams {
                speed: 0.25,
                zoom: 3.0,
                trail: 0.15,
                flame_brightness: 0.2,
            },
            kifs: KifsParams {
                fold_angle: 0.0,
                scale: 0.0,
                brightness: 0.0,
            },
            transforms: self.transforms,
            final_transform: self.final_transform,
            symmetry: self.symmetry,
            palette: self.palette,
            parent_a: None,
            parent_b: None,
            generation: 0,
        }
    }
}

fn parse_xform(e: &quick_xml::events::BytesStart) -> FlameTransform {
    let mut xf = FlameTransform {
        weight: 0.0, // will be set from attribute
        ..Default::default()
    };

    for attr in e.attributes().flatten() {
        let key = String::from_utf8_lossy(attr.key.as_ref()).to_string();
        let val = String::from_utf8_lossy(&attr.value).to_string();

        match key.as_str() {
            "weight" => xf.weight = val.parse().unwrap_or(1.0),
            "color" => xf.color = val.parse().unwrap_or(0.0),
            "coefs" => {
                let nums: Vec<f32> = val
                    .split_whitespace()
                    .filter_map(|s| s.parse().ok())
                    .collect();
                if nums.len() >= 6 {
                    // Flam3 coefs="a d b e c f" (column-major)
                    // Map to our row-major: a=coefs[0], b=coefs[2], c=coefs[1], d=coefs[3]
                    xf.affine[0][0] = nums[0]; // flam3 a
                    xf.affine[0][1] = nums[2]; // flam3 b
                    xf.affine[1][0] = nums[1]; // flam3 d
                    xf.affine[1][1] = nums[3]; // flam3 e
                    xf.offset[0] = nums[4]; // flam3 c (translation x)
                    xf.offset[1] = nums[5]; // flam3 f (translation y)
                }
            }
            other => {
                // Check if it's a known variation name
                if let Some(_idx) = VARIATION_NAMES.iter().position(|&name| name == other) {
                    let v: f32 = val.parse().unwrap_or(0.0);
                    set_variation_by_name(&mut xf, other, v);
                } else if VARIATION_PARAM_NAMES.contains(&other) {
                    // Parametric variation parameter
                    let v: f32 = val.parse().unwrap_or(0.0);
                    xf.variation_params.insert(other.to_string(), v);
                } else if ![
                    "symmetry",
                    "opacity",
                    "animate",
                    "motion_frequency",
                    "motion_function",
                    "var_color",
                    "post",
                    "chaos",
                ]
                .contains(&other)
                {
                    // Log unsupported variations (skip known non-variation attrs)
                    if !other.starts_with("pre_") && !other.starts_with("post_") {
                        eprintln!("[flam3] unsupported attribute: {other}={val}");
                    } else {
                        eprintln!("[flam3] unsupported variation: {other}");
                    }
                }
            }
        }
    }

    // Default weight if not specified
    if xf.weight <= 0.0 {
        xf.weight = 1.0;
    }

    xf
}

fn set_variation_by_name(xf: &mut FlameTransform, name: &str, val: f32) {
    match name {
        "linear" => xf.linear = val,
        "sinusoidal" => xf.sinusoidal = val,
        "spherical" => xf.spherical = val,
        "swirl" => xf.swirl = val,
        "horseshoe" => xf.horseshoe = val,
        "handkerchief" => xf.handkerchief = val,
        "julia" => xf.julia = val,
        "polar" => xf.polar = val,
        "disc" => xf.disc = val,
        "rings" => xf.rings = val,
        "bubble" => xf.bubble = val,
        "fisheye" => xf.fisheye = val,
        "exponential" => xf.exponential = val,
        "spiral" => xf.spiral = val,
        "diamond" => xf.diamond = val,
        "bent" => xf.bent = val,
        "waves" => xf.waves = val,
        "popcorn" => xf.popcorn = val,
        "fan" => xf.fan = val,
        "eyefish" => xf.eyefish = val,
        "cross" => xf.cross = val,
        "tangent" => xf.tangent = val,
        "cosine" => xf.cosine = val,
        "blob" => xf.blob = val,
        "noise" => xf.noise = val,
        "curl" => xf.curl = val,
        _ => eprintln!("[flam3] unknown variation: {name}"),
    }
}

fn parse_palette(hex: &str) -> Vec<[f32; 3]> {
    let clean: String = hex.chars().filter(|c| c.is_ascii_hexdigit()).collect();
    clean
        .as_bytes()
        .chunks(6)
        .filter_map(|chunk| {
            let hex_str = std::str::from_utf8(chunk).ok()?;
            if hex_str.len() < 6 {
                return None;
            }
            let r = u8::from_str_radix(&hex_str[0..2], 16).unwrap_or(0) as f32 / 255.0;
            let g = u8::from_str_radix(&hex_str[2..4], 16).unwrap_or(0) as f32 / 255.0;
            let b = u8::from_str_radix(&hex_str[4..6], 16).unwrap_or(0) as f32 / 255.0;
            Some([r, g, b])
        })
        .collect()
}

pub fn load_random_flame(dir: &Path) -> Result<FlameGenome, String> {
    let entries: Vec<_> = fs::read_dir(dir)
        .map_err(|e| format!("read dir {}: {e}", dir.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|ext| ext == "flame" || ext == "xml")
        })
        .collect();

    if entries.is_empty() {
        return Err(format!("no .flame files found in {}", dir.display()));
    }

    let entry = entries.choose(&mut rand::rng()).ok_or("empty")?;
    let path = entry.path();
    let xml = fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;

    let flam3 = Flam3File::parse(&xml)?;
    if flam3.flames.is_empty() {
        return Err(format!("no flames in {}", path.display()));
    }

    // Pick a random flame from the file
    let flame = flam3.flames.into_iter().collect::<Vec<_>>();
    let idx = rand::rng().random_range(0..flame.len());
    Ok(flame.into_iter().nth(idx).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_FLAME: &str = r#"<flames>
<flame name="test_flame" symmetry="2">
    <xform weight="0.5" color="0.0" coefs="1 0 0 1 0 0" spherical="1.0"/>
    <xform weight="0.5" color="0.5" coefs="0.5 0.5 -0.5 0.5 0.1 0.2" linear="0.6" sinusoidal="0.4"/>
</flame>
</flames>"#;

    #[test]
    fn parse_minimal_flame() {
        let file = Flam3File::parse(MINIMAL_FLAME).unwrap();
        assert_eq!(file.flames.len(), 1);
        let flame = &file.flames[0];
        assert_eq!(flame.name, "test_flame");
        assert_eq!(flame.symmetry, 2);
    }

    #[test]
    fn parse_transforms() {
        let file = Flam3File::parse(MINIMAL_FLAME).unwrap();
        let flame = &file.flames[0];
        assert_eq!(flame.transforms.len(), 2);
        assert_eq!(flame.transforms[0].spherical, 1.0);
        assert_eq!(flame.transforms[0].weight, 0.5);
        assert_eq!(flame.transforms[0].color, 0.0);
        assert!((flame.transforms[1].linear - 0.6).abs() < 0.01);
        assert!((flame.transforms[1].sinusoidal - 0.4).abs() < 0.01);
    }

    #[test]
    fn parse_affine_coefs() {
        let file = Flam3File::parse(MINIMAL_FLAME).unwrap();
        let xf = &file.flames[0].transforms[0];
        // coefs="1 0 0 1 0 0" → identity affine
        // flam3 layout: a d b e c f (column-major)
        // Mapped: xf.a() = nums[0]=1, xf.d() = nums[3]=1
        assert!((xf.a() - 1.0).abs() < 0.01);
        assert!((xf.d() - 1.0).abs() < 0.01);
        assert!((xf.b() - 0.0).abs() < 0.01);
        assert!((xf.c() - 0.0).abs() < 0.01);
    }

    #[test]
    fn parse_palette_hex() {
        let xml = r#"<flames>
<flame name="pal_test">
    <xform weight="1.0" color="0.0" coefs="1 0 0 1 0 0" linear="1.0"/>
    <palette count="256">
FF0000FF000000FF00
    </palette>
</flame>
</flames>"#;
        let file = Flam3File::parse(xml).unwrap();
        let flame = &file.flames[0];
        assert!(flame.palette.is_some());
        let palette = flame.palette.as_ref().unwrap();
        assert!(!palette.is_empty());
        // First color should be red (FF0000)
        assert!(
            (palette[0][0] - 1.0).abs() < 0.01,
            "red channel should be 1.0, got {}",
            palette[0][0]
        );
        assert!(
            (palette[0][1] - 0.0).abs() < 0.01,
            "green channel should be 0.0"
        );
    }

    #[test]
    fn parse_empty_xml() {
        let file = Flam3File::parse("<flames></flames>").unwrap();
        assert!(file.flames.is_empty());
    }

    #[test]
    fn parse_lineage_defaults() {
        let file = Flam3File::parse(MINIMAL_FLAME).unwrap();
        let flame = &file.flames[0];
        assert!(flame.parent_a.is_none());
        assert!(flame.parent_b.is_none());
        assert_eq!(flame.generation, 0);
    }
}

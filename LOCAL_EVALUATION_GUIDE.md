# Local Evaluation Guide

## Perubahan yang Dilakukan

File `utils/run_evaluation.py` telah dimodifikasi untuk mendukung **pembacaan file JSON lokal** sebagai alternatif dari pemanggilan API.

## Cara Penggunaan

### 1. **Menggunakan API (Cara Lama)**
```bash
python -m utils.run_evaluation --task_id 9fb20acd-836b-4450-8e64-4bfba602b6b6 --models <repo>
```

### 2. **Menggunakan File JSON Lokal (Cara Baru)**

#### Dengan relative path:
```bash
python -m utils.run_evaluation --task_id contoh.json --models <repo>
```

#### Dengan absolute path:
```bash
python -m utils.run_evaluation --task_id /Users/firzahadzami/Documents/GRADIENT/G.O.D/contoh.json --models <repo>
```

## Cara Kerja

Kode akan otomatis mendeteksi apakah `--task_id` adalah:
- **File JSON** (berakhiran `.json`)
- **Path yang valid** (file exists)
- **Task ID biasa** (akan fetch dari API)

### Contoh Lengkap:
```bash
# Evaluasi model Anda menggunakan task data dari contoh.json
python -m utils.run_evaluation \
  --task_id contoh.json \
  --models your-username/your-trained-model \
  --gpu_ids 0
```

## Keuntungan

✅ **Offline Mode**: Tidak perlu koneksi internet atau API  
✅ **Testing**: Mudah untuk testing dengan data lokal  
✅ **Reproducibility**: Data task tersimpan permanen di file lokal  
✅ **Flexibility**: Bisa edit JSON untuk custom testing scenarios  

## Format JSON

File JSON harus memiliki struktur yang sama dengan response API, minimal berisi:
- `task_type`: Tipe task (ImageTask, InstructTextTask, dll)
- `model_id`: Model base yang digunakan
- `test_data`: URL atau path ke test dataset
- `model_type`: Tipe model (sdxl, flux, dll) untuk ImageTask
- `hotkey_details`: (Optional) List model yang akan dievaluasi

Lihat `contoh.json` untuk referensi lengkap.

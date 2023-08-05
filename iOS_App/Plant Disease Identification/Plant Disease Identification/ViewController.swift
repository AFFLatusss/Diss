//
//  ViewController.swift
//  pytorchOnnx
//
//  Created by 余荣键 on 20/07/2023.
//

import Foundation
import UIKit
import CoreML
import CoreImage

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
//    @IBOutlet weak var selectImageButton: UIButton!
//    @IBOutlet weak var imageView: UIImageView!
    
//    @IBAction func selectImageTapped(_ sender: UIButton) {
//        let imagePicker = UIImagePickerController()
//        imagePicker.sourceType = .photoLibrary
//        imagePicker.delegate = self
//        present(imagePicker, animated: true, completion: nil)
//    }
    
    private let label:UILabel = {
        let label = UILabel()
        label.textAlignment = .center
        label.numberOfLines = 0
        label.text = "Predictions will show here.."
        label.font = label.font.withSize(20)
        return label
    }()
    
    private let imageView:UIImageView = {
        let imageView = UIImageView()
        imageView.image = UIImage(systemName:  "leaf.fill")
        imageView.contentMode = .scaleAspectFit
        imageView.isUserInteractionEnabled = true
        imageView.tintColor = .systemGreen
    
         
        return imageView
    }()
//    let model = ModelManager.shared.getModel()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        view.addSubview(label)
        view.addSubview(imageView)
        
        title = "Plant Disease Identification"
        
        let tap = UITapGestureRecognizer(target: self, action: #selector(didTapImage))
        tap.numberOfTapsRequired = 1
        imageView.addGestureRecognizer(tap)
    }
    
    @objc func didTapImage(){
        let picker = UIImagePickerController()
        picker.sourceType = .photoLibrary
        picker.delegate = self
        present(picker,  animated: true )
    }
 
    override func viewDidLayoutSubviews () {
        super.viewDidLayoutSubviews()
        
        imageView.frame = CGRect(x:20, y:view.safeAreaInsets.top,
                                 width:view.frame.size.width-40,
                                 height: view.frame.size.height-400)
        
//        label.frame = CGRect(x:20, y:view.safeAreaInsets.top+(view.frame.size.width-40)+10,
//                             width: view.frame.size.width-40,
//                             height:100)
        label.frame = CGRect(x: 20, y: view.frame.size.height - 200,
                             width: view.frame.size.width - 40,
                             height: 100)
    }
    
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        picker.dismiss(animated: true, completion: nil)
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        
        picker.dismiss(animated: true, completion: nil)

        if let image = info[.originalImage] as? UIImage {
            imageView.image = image
            // Perform the prediction using the selected image
            predictImage(image)
        }
    }
    
    
    // MARK: - Image preprocessing and prediction

    func predictImage(_ image: UIImage) {

        guard let pixelBufferInput = convertImage(image: image)else {
            return
        }
        
        
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .cpuOnly
            
            let model = try combined_model(configuration: config)
            let input = combined_modelInput(x_1: pixelBufferInput)
            
            let output = try model.prediction(input: input)
            let text = output.classLabel
            
            print("output is ",output.var_824)
            label.text = text
            
            print("Predicted Label: \(text)")
            
        } catch {
            print(error.localizedDescription)
        }
        
    }
}

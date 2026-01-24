import { motion } from 'framer-motion'
import { Brain, Target, Users, Shield, Zap, Award, CheckCircle, GitBranch } from 'lucide-react'

const About = () => {
  const features = [
    {
      icon: <Brain className="h-8 w-8 text-primary-600" />,
      title: "Advanced AI Models",
      description: "Our platform employs four cutting-edge AI models: CNN for pattern recognition, RNN for deep learning analysis, LSTM for mood detection, and BERT for comprehensive assessment."
    },
    {
      icon: <Target className="h-8 w-8 text-green-600" />,
      title: "High Accuracy",
      description: "Achieving 95%+ accuracy in mental health detection through ensemble learning and advanced preprocessing techniques."
    },
    {
      icon: <Shield className="h-8 w-8 text-blue-600" />,
      title: "Privacy & Security",
      description: "Your data is processed with the highest standards of privacy and security. All analyses are performed locally and securely."
    },
    {
      icon: <Zap className="h-8 w-8 text-yellow-600" />,
      title: "Real-time Analysis",
      description: "Get instant results with our optimized processing pipeline that delivers comprehensive reports in seconds."
    }
  ]

  const models = [
    {
      name: "CNN",
      fullName: "Convolutional Neural Network",
      description: "Specialized in pattern recognition and feature extraction from text data, identifying complex linguistic patterns indicative of mental health states.",
      accuracy: "",
      focus: "Pattern Recognition"
    },
    {
      name: "RNN",
      fullName: "Deep Neural Network",
      description: "Multi-layered neural architecture for deep learning analysis, capturing intricate relationships in language and sentiment patterns.",
      accuracy: "",
      focus: "Deep Learning"
    },
    {
      name: "LSTM",
      fullName: "Mood-Oriented Neural",
      description: "Specialized neural network designed specifically for mood detection and emotional state analysis from textual content.",
      accuracy: "",
      focus: "Mood Detection"
    },
    {
      name: "BERT",
      fullName: "Comprehensive Assessment System for Text-based Life Evaluation",
      description: "Holistic evaluation system that combines multiple assessment techniques for comprehensive mental health evaluation.",
      accuracy: "",
      focus: "Comprehensive Assessment"
    }
  ]

  const analysisTypes = [
    {
      title: "Sentiment Analysis",
      description: "Classifies text into positive, negative, and neutral sentiments",
      metrics: ["Polarity Score", "Subjectivity Analysis", "Confidence Rating"]
    },
    {
      title: "Risk Assessment",
      description: "Evaluates potential mental health risks on low, medium, and high scales",
      metrics: ["Risk Probability", "Severity Index", "Intervention Priority"]
    },
    {
      title: "Emotion Detection",
      description: "Identifies specific emotions including joy, sadness, anger, fear, surprise, and disgust",
      metrics: ["Emotion Intensity", "Emotional Variance", "Dominant Emotions"]
    }
  ]

  const teamStats = [
    { label: "AI Researchers", value: "" },
    { label: "Mental Health Experts", value: "" },
    { label: "Data Scientists", value: "" },
    { label: "Software Engineers", value: "" }
  ]

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Hero Section */}
      <section className="bg-gradient-to-br from-primary-600 to-purple-700 text-white py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center"
          >
            <h1 className="text-4xl md:text-5xl font-bold mb-6">
              About Mental Health Detector
            </h1>
            <p className="text-xl md:text-2xl mb-8 max-w-4xl mx-auto text-primary-100">
              Revolutionizing mental health detection through advanced AI and machine learning, 
              providing accessible and accurate mental health insights from social media and text data.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Mission Section */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-6">
              Our Mission
            </h2>
            <p className="text-xl text-gray-600 max-w-4xl mx-auto">
              To democratize mental health awareness and early detection through cutting-edge AI technology, 
              making mental health assessment more accessible, accurate, and timely for individuals and 
              healthcare professionals worldwide.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="card text-center"
              >
                <div className="mb-4 flex justify-center">{feature.icon}</div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">
                  {feature.title}
                </h3>
                <p className="text-gray-600">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* AI Models Section */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-6">
              AI Models & Technology
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Our platform leverages four specialized AI models, each designed for specific aspects 
              of mental health detection and analysis.
            </p>
          </motion.div>

          <div className="grid lg:grid-cols-2 gap-8">
            {models.map((model, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: index % 2 === 0 ? -30 : 30 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="card"
              >
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <h3 className="text-2xl font-bold text-primary-600 mb-1">
                      {model.name}
                    </h3>
                    <p className="text-lg font-medium text-gray-700">
                      {model.fullName}
                    </p>
                  </div>
                  <div className="text-right">
                    <div className="text-2xl font-bold text-green-600">
                      {model.accuracy}
                    </div>
                    <div className="text-sm text-gray-500"></div>
                  </div>
                </div>
                
                <p className="text-gray-600 mb-4">{model.description}</p>
                
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-primary-600 bg-primary-50 px-3 py-1 rounded-full">
                    {model.focus}
                  </span>
                  <CheckCircle className="h-5 w-5 text-green-500" />
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Analysis Types Section */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-6">
              Comprehensive Analysis Types
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Our platform provides three core types of mental health analysis, 
              each offering detailed insights and professional-grade metrics.
            </p>
          </motion.div>

          <div className="grid lg:grid-cols-3 gap-8">
            {analysisTypes.map((type, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="card"
              >
                <h3 className="text-xl font-semibold text-gray-900 mb-4">
                  {type.title}
                </h3>
                <p className="text-gray-600 mb-6">{type.description}</p>
                
                <div className="space-y-2">
                  <h4 className="font-medium text-gray-900">Key Metrics:</h4>
                  {type.metrics.map((metric, metricIndex) => (
                    <div key={metricIndex} className="flex items-center">
                      <CheckCircle className="h-4 w-4 text-green-500 mr-2 flex-shrink-0" />
                      <span className="text-sm text-gray-600">{metric}</span>
                    </div>
                  ))}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Team Stats Section */}
      <section className="py-20 bg-primary-600 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold mb-6">
              Expert Team
            </h2>
            <p className="text-xl text-primary-100 max-w-3xl mx-auto">
              Our multidisciplinary team combines expertise in AI, mental health, 
              and software engineering to deliver cutting-edge solutions.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {teamStats.map((stat, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, scale: 0.8 }}
                whileInView={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                className="text-center"
              >
                <div className="text-4xl md:text-5xl font-bold text-yellow-300 mb-2">
                  {stat.value}
                </div>
                <div className="text-primary-100 font-medium">{stat.label}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Technology Stack Section */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-6">
              Technology Stack
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Built with modern technologies and frameworks to ensure scalability, 
              performance, and reliability.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            <div className="card text-center">
              <GitBranch className="h-12 w-12 text-blue-600 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Frontend</h3>
              <p className="text-gray-600 text-sm">React, TypeScript, Tailwind CSS</p>
            </div>
            <div className="card text-center">
              <Brain className="h-12 w-12 text-green-600 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-gray-900 mb-2">AI/ML</h3>
              <p className="text-gray-600 text-sm">TensorFlow, PyTorch, scikit-learn</p>
            </div>
            <div className="card text-center">
              <Shield className="h-12 w-12 text-purple-600 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Security</h3>
              <p className="text-gray-600 text-sm">End-to-end encryption, GDPR compliant</p>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}

export default About

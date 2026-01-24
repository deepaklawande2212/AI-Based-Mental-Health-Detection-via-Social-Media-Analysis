import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Brain, Shield, BarChart3, Users, ArrowRight } from 'lucide-react'

const Home = () => {
  const features = [
    {
      icon: <Brain className="h-8 w-8 text-primary-600" />,
      title: "AI-Powered Analysis",
      description: "Advanced machine learning models including CNN, DNN, MOON, and CASTLE for accurate mental health detection."
    },
    {
      icon: <BarChart3 className="h-8 w-8 text-mental-green" />,
      title: "Comprehensive Reports",
      description: "Get detailed sentiment analysis, risk assessment, and emotion detection with actionable insights."
    },
    {
      icon: <Shield className="h-8 w-8 text-mental-red" />,
      title: "Privacy First",
      description: "Your data is secure and processed with the highest standards of privacy and confidentiality."
    },
    {
      icon: <Users className="h-8 w-8 text-mental-purple" />,
      title: "Multi-Platform Support",
      description: "Analyze text from CSV files or directly from Twitter/social media platforms."
    }
  ]

  const stats = [
    { number: "95%", label: "Accuracy Rate" },
    { number: "100", label: "Analyses Completed" },
    { number: "4", label: "AI Models Used" },
    { number: "24/7", label: "Available" }
  ]

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="gradient-bg text-white py-20 relative overflow-hidden">
        <div className="absolute inset-0 bg-black/20"></div>
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center"
          >
            <h1 className="text-4xl md:text-6xl font-bold mb-6">
              Mental Health Detection
              <span className="block text-yellow-300">Made Simple</span>
            </h1>
            <p className="text-xl md:text-2xl mb-8 max-w-3xl mx-auto text-gray-100">
              Advanced AI-powered platform to detect and analyze mental health patterns 
              from social media content and text data with professional-grade accuracy.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link to="/dashboard" className="btn-primary inline-flex items-center">
                Start Analysis <ArrowRight className="ml-2 h-5 w-5" />
              </Link>
              <Link to="/about" className="btn-secondary inline-flex items-center">
                Learn More
              </Link>
            </div>
          </motion.div>
        </div>
        
        {/* Floating Elements */}
        <div className="absolute top-20 left-10 animate-pulse-slow">
          <div className="w-4 h-4 bg-yellow-300 rounded-full"></div>
        </div>
        <div className="absolute top-40 right-20 animate-pulse-slow delay-300">
          <div className="w-6 h-6 bg-pink-300 rounded-full"></div>
        </div>
        <div className="absolute bottom-20 left-1/4 animate-pulse-slow delay-700">
          <div className="w-3 h-3 bg-blue-300 rounded-full"></div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, scale: 0.8 }}
                whileInView={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                className="text-center"
              >
                <div className="text-3xl md:text-4xl font-bold text-primary-600 mb-2">
                  {stat.number}
                </div>
                <div className="text-gray-600 font-medium">{stat.label}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              Powerful Features for Mental Health Analysis
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Our platform combines cutting-edge AI technology with user-friendly interfaces 
              to provide comprehensive mental health insights.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="card hover:shadow-xl transition-shadow duration-300"
              >
                <div className="mb-4">{feature.icon}</div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">
                  {feature.title}
                </h3>
                <p className="text-gray-600">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-primary-600 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-3xl md:text-4xl font-bold mb-6">
              Ready to Get Started?
            </h2>
            <p className="text-xl mb-8 max-w-2xl mx-auto text-primary-100">
              Join thousands of professionals using our platform for mental health analysis. 
              Start your first analysis today.
            </p>
            <Link 
              to="/dashboard" 
              className="bg-white text-primary-600 hover:bg-gray-100 px-8 py-4 rounded-lg font-semibold text-lg transition-colors duration-200 shadow-lg hover:shadow-xl inline-flex items-center"
            >
              Start Free Analysis <ArrowRight className="ml-2 h-5 w-5" />
            </Link>
          </motion.div>
        </div>
      </section>
    </div>
  )
}

export default Home
